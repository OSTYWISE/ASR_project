import editdistance


def calc_cer(target_text, predicted_text) -> float:
    if not target_text:
        if predicted_text:
            return 1
        return 0
    return editdistance.eval(target_text.split(), predicted_text.split()) / len(target_text.split())


def calc_wer(target_text: str, predicted_text: str) -> float:
    if not target_text:
        if predicted_text:
            return 1
        return 0
    return editdistance.eval(target_text, predicted_text) / len(target_text)
