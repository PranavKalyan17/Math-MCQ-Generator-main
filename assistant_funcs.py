import json
from time import sleep
from openai import OpenAI
from assistant_configs import *


def create_generator_model(client: OpenAI, grade_level: int) -> str:
    """Returns the assistant.id of the new assistant."""

    params = get_generator_params(grade_level=grade_level)
    return _get_assistant_id(client, params)
    # return "asst_Q6dkuUYESiS0OGMgufgJYMz4"


def create_validator_model(client: OpenAI, grade_level: int) -> str:
    """Returns the assistant.id of the new assistant."""

    params = get_validator_params(grade_level=grade_level)
    return _get_assistant_id(client, params)
    # return "asst_wafxg6ULdPPUkQpJomCcpjAo"


def _get_assistant_id(client: OpenAI, assistant_params: dict):
    existing_assistants = client.beta.assistants.list()
    # with open("debugging/existing_assistants.text", "w") as f:
    #     f.write(str(existing_assistants.data[0].__dict__))
    #     f.close()

    for ass in existing_assistants.data:
        if (
            ass.name == assistant_params["name"]
            and ass.instructions == assistant_params["instructions"]
            # and
        ):
            return ass.id

    new_assistant = client.beta.assistants.create(**assistant_params)
    return new_assistant.id


def create_thread(client: OpenAI, user_message: str, file_id: str = None) -> str:
    thread = client.beta.threads.create()

    if file_id is not None:
        thread_message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_message,
            file_ids=[file_id],
        )
    else:
        thread_message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_message,
        )

    return thread.id


def format_math_expressions(
    client: OpenAI, mcq_str: str, grade_level: int, retry_count: int = 0
) -> str:
    try:
        formatter_params = get_formatter_params(grade_level=grade_level)

        response = client.chat.completions.create(
            model=formatter_params["model"],
            # response_format=FORMATTER_RESPONSE_FORMAT,
            tools=formatter_params["tools"],
            tool_choice=formatter_params["tool_choice"],
            messages=[
                {"role": "system", "content": formatter_params["instructions"]},
                {"role": "user", "content": f"{mcq_str}"},
            ],
        )

        new_json_str = response.choices[0].message.tool_calls[0].function.arguments
    except Exception as e:
        print("Format Checker Completions call failed.")
        print(e)

    with open("debugging/mcq_str.txt", "w") as f:
        f.write(mcq_str)
        f.close()

    with open("debugging/new_json_str.txt", "w") as f:
        f.write(new_json_str)
        f.close()

    try:
        new_json = json.loads(new_json_str)

    except Exception as e:
        print(f"Formatter failed. Failed count: {retry_count}")
        print(e)
        if retry_count >= MAX_RUN_RETRIES:
            # if max retires reached, return an empty json string
            return "{}"

        return format_math_expressions(
            client,
            mcq_str=mcq_str,
            grade_level=grade_level,
            retry_count=retry_count + 1,
        )

    return new_json_str


def create_and_excute_run(
    client: OpenAI,
    thread_id: str,
    assistant_id: str,
    refinement_message: str = None,
    retry_count: int = 0,
):
    if refinement_message:
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=refinement_message,
        )

    # create new run
    run = client.beta.threads.runs.create(
        thread_id=thread_id, assistant_id=assistant_id
    )
    run_failed = False
    tool_called = False

    while run.status != "completed":
        print(f"> RUN STATUS: {run.status}")
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)

        if run.status == "failed" or run.status == "expired":
            print("> RUN STATUS: {run.status}")
            run_failed = True
            break

        if (
            run.status == "requires_action"
            and run.required_action.type == "submit_tool_outputs"
        ):
            print("> calling tools...")
            tool_calls = run.required_action.submit_tool_outputs.tool_calls
            tool_outputs = [
                {"tool_call_id": tool_call.id, "output": tool_call.function.arguments}
                for tool_call in tool_calls
            ]
            client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread_id, run_id=run.id, tool_outputs=tool_outputs
            )
            tool_called = True

        sleep(2)

    # get messages after run has finished
    messages = client.beta.threads.messages.list(thread_id=thread_id)

    # if run fails, rerun it up to 3 times
    if run_failed or not tool_called:
        print("> Run failed.")

        if retry_count >= MAX_RUN_RETRIES:
            print("> Max retries reached.")
            return [], {}
        else:
            print("> Reattempting run...")
            return create_and_excute_run(
                client,
                thread_id,
                assistant_id,
                retry_count=retry_count + 1,
            )
    else:
        print("> Run completed.")
        return messages, tool_outputs[-1]["output"]


def run_full_thread(
    client: OpenAI,
    assistant_id: str,
    validator_id: str,
    # inputs
    core_inputs_message: str,
    file_id: str,
    grade_level: int,
):
    # validate output
    is_valid = False
    val_thread_ids = []
    validations_count = 0

    try:
        # Generate question
        gen_thread_id = create_thread(
            client=client,
            user_message=core_inputs_message,
            file_id=file_id,
        )
        messages, output = create_and_excute_run(client, gen_thread_id, assistant_id)

        with open("debugging/generator_output.json", "w") as f:
            f.write(output)

        # Validate question
        while 0 <= validations_count and validations_count < VALIDATOR_MAX_RETRIES:
            validations_count += 1
            print(f"Validating quesiton, attempt number {validations_count}...")

            try:
                val_thread_id = create_thread(client=client, user_message=output)
                val_thread_ids.append(val_thread_id)
                _, validator_output = create_and_excute_run(
                    client, val_thread_id, validator_id
                )

                with open("debugging/validator_output.json", "w") as f:
                    f.write(validator_output)

                correctness_outputs_json = json.loads(validator_output)
                is_valid = correctness_outputs_json["correct"]

                if correctness_outputs_json["run_fail"]:
                    correction_message = "The question might be broken, use code interpreter to verify the mathematical validity and fix any errors. Output in the same format"  # what is this?
                else:
                    correction_message = f"Given the following reasoning fix the generated question and output in the same format:'{str(correctness_outputs_json['explanation'])}'"

            except Exception as e:
                print("Validator run failed.")
                print(e)
                correction_message = "The question might be broken, use code interpreter to verify the mathematical validity and fix any errors. Output in the same format"
                # validations_count = -2

            if is_valid:
                break

            messages, output = create_and_excute_run(
                client,
                gen_thread_id,
                assistant_id,
                refinement_message=correction_message,
            )
            # validations_count = validations_count + 1

    except Exception as e:
        print("Validator Error.")
        print(e)
        messages, output = [], {}

    if validations_count == VALIDATOR_MAX_RETRIES and not is_valid:
        print(f"Max validation retries reached. Question is not valid.")
        validations_count = VALIDATOR_MAX_RETRIES + 1

    try:
        json.loads(output)
    except Exception as e:
        print(f"output: {output}")
        print(e)
        correction_message = "Use code interpreter to validate and fix all errors in the output json and return the fixed json output"
        try:
            messages, output = create_and_excute_run(
                client,
                gen_thread_id,
                assistant_id,
                refinement_message=correction_message,
            )
        except Exception as e:
            print(e)
            messages, output = [], {}

    formatter_worked = False
    # run format checker to convert math expressions to LaTeX
    if messages != [] and output != {}:
        # print(output)
        with open("debugging/ass_output.txt", "w") as f:
            f.write(output)

        try:
            ass_output_json = json.loads(output)
            plain_text = (
                f"Question: {ass_output_json['question']['text_with_inline_latex']}"
            )

            answer_letters = ["A", "B"]

            if "C" in ass_output_json.keys():
                answer_letters.append("C")
            if "D" in ass_output_json.keys():
                answer_letters.append("D")

            # print(f"answer_letters: {answer_letters}")

            for c in answer_letters:
                plain_text += f"\n\nAnswer {c}: {ass_output_json[c]['answer']['text_with_inline_latex']}\n"
                plain_text += (
                    f"Answer {c} is Correct: {ass_output_json[c]['correct']}\n"
                )
                plain_text += f"Explanation {c}: {ass_output_json[c]['explanation']['text_with_inline_latex']}"

            # print(f"plain_text: {plain_text}")

            output = format_math_expressions(
                client=client,
                mcq_str=plain_text,
                grade_level=grade_level,
            )
            formatter_worked = True

            with open("debugging/formatter_output.json", "w") as f:
                f.write(output)

        except Exception as e:
            print("JSON parsing for Formatting failed. Format Chekcer not used.")
            print(e)

            return messages, output, gen_thread_id, validations_count, val_thread_ids

    if formatter_worked:
        print(f"Question Generation Successful.")

    return messages, output, gen_thread_id, validations_count, val_thread_ids
