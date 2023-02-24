import torch


class Accuracy:
    """
    Classification accuracy (top-1). This class records per-batch accuracy that
    can be retrieved at the end of evaluation.

    When using with DDP, user must aggregate results across GPU processes.
    """

    def __init__(self, num_classes: int, mean_per_class: bool = False):
        """
        Args:
            num_classes: Number of classes in evaluation dataset.
            mean_per_class: Whether to use mean per-class accuracy. This flag is
                useful when evaluation set has class imbalance; redundant in a
                class-balanced scenario.
        """
        self.num_classes = num_classes

        # Collect per-class correct/total examples to calculate accuracy.
        self.correct_per_class = torch.zeros(num_classes)
        self.total_per_class = torch.zeros(num_classes)
        self.mean_per_class = mean_per_class

    def reset(self):
        self.correct_per_class.zero_()
        self.total_per_class.zero_()

    def __call__(self, predictions: torch.Tensor, ground_truth: torch.Tensor):
        """
        Record the accuracy of current batch of predictions and ground-truth.

        Args:
            predictions: Predicted logits or probabilities. Tensor of shape
                ``(batch_size, num_classes)``.
            ground_truth: Ground-truth integer labels, tensor of shape
                ``(batch_size, )`` with values in ``[0, num_classes-1]``.

        Returns:
            Accuracy (in percentage) so far.
        """

        # shape: (batch_size, )
        pred_id = predictions.argmax(dim=-1)
        is_correct = (pred_id == ground_truth).cpu()
        ground_truth = ground_truth.cpu()

        # FIXME: `torch.unique` is causing RuntimeError. Retrying it can avoid
        # the issue -- investigate this later.
        try:
            unique_gt_ids = ground_truth.unique()
        except RuntimeError:
            unique_gt_ids = ground_truth.clone().unique()

        for gt_id in unique_gt_ids:
            self.correct_per_class[gt_id] += is_correct[ground_truth == gt_id].sum()
            self.total_per_class[gt_id] += (ground_truth == gt_id).sum()

        return self.result

    @property
    def result(self) -> float:
        # Regular accuracy or mean per-class accuracy (avoid zero division).
        if self.mean_per_class:
            acc_per_class = self.correct_per_class / (self.total_per_class + 1e-12)
            _result = acc_per_class.mean()
        else:
            num_correct = self.correct_per_class.sum()
            num_total = self.total_per_class.sum()
            _result = num_correct / (num_total + 1e-12)

        return _result.item() * 100.0
