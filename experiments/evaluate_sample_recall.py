def evaluate_sample_recall(model, samples):
    """
    Evaluate CDSN recall over every individual sample.

    Returns:
        visual_recall:
            shape + color -> correct name

        auditory_recall:
            name -> valid associated visual representation
    """

    visual_correct = 0
    auditory_correct = 0

    total = len(samples)

    for sample in samples:

        name = sample["name"]
        shape = sample["shape"]
        color = sample["color"]

        # --------------------------------
        # Visual -> auditory/name recall
        # --------------------------------
        visual_result = model.recall_name_from_visual(
            shape_vector=shape,
            color_vector=color
        )

        if name in visual_result["names"]:
            visual_correct += 1

        # --------------------------------
        # Auditory/name -> visual recall
        # --------------------------------
        auditory_result = model.recall_visual_from_name(
            name
        )

        if auditory_result is not None:
            auditory_correct += 1

    return {
        "visual_correct": visual_correct,
        "auditory_correct": auditory_correct,
        "total_samples": total,

        "visual_recall":
            visual_correct / total if total else 0.0,

        "auditory_recall":
            auditory_correct / total if total else 0.0,
    }