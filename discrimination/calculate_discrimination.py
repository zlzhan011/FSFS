import copy


def calculate_original_discrimination(class_label, discrimination_column):
    assert len(class_label) == len(
        discrimination_column), "the length of class_label and discrimination_column isn't equal"
    deprived_rejected = 0
    deprived_granted = 0
    favored_rejected = 0
    favored_granted = 0
    for i in range(len(discrimination_column)):
        if discrimination_column[i] == 0:
            if class_label[i] == 0:
                deprived_rejected = deprived_rejected + 1
            else:
                deprived_granted = deprived_granted + 1
        else:
            if class_label[i] == 0:
                favored_rejected = favored_rejected + 1
            else:
                favored_granted = favored_granted + 1

    print("favored_granted:", favored_granted, "--favored_rejected:", favored_rejected, "--deprived_granted:", deprived_granted, "--deprived_rejected:", deprived_rejected)
    discrimination_score = (favored_granted / (favored_granted + favored_rejected)) - (deprived_granted / (
                deprived_granted + deprived_rejected))

    return discrimination_score
