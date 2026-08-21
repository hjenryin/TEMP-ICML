def extract_answer(answer):
    # Try to extract content inside \boxed{}
    answer = remove_boxed(last_boxed_only_string(answer))
    answer = remove_boxed(last_boxed_only_string(answer, "\\text"), "\\text")
    return answer


def remove_boxed(s, left="\\boxed"):
    original_s = s
    # NOTE: Need to append "{"
    left = left + "{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        answer = s[len(left) : -1]
        if "=" in answer:
            answer = answer.split("=")[-1].lstrip(" ")
        return answer
    except Exception:
        return original_s


def last_boxed_only_string(string, left="\\boxed"):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return string
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = string
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval
