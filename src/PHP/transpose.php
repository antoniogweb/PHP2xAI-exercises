<?php

use PHP2xAI\Runtime\PHP\Core\GraphRuntime;
use PHP2xAI\Tensor\Tensor;

include("../../vendor/autoload.php");

function runTranspose(string $name, Tensor $input, ?array $axes = null): void
{
	$input->setRequiresGrad(true);
	$output = $axes === null ? $input->transpose() : $input->transpose($axes);
	$runtime = GraphRuntime::createFromOutputTensor($output);
	$runtime->forward();
	$runtime->backward();
	$runtime->refreshTensorsData();

	echo "\n=== {$name} ===\n";
	echo "input shape:  [" . implode(', ', $input->shape) . "]\n";
	echo "output shape: [" . implode(', ', $output->shape) . "]\n";
	$output->printData();
	$input->printGrad();
}

// Fast path: TRANSPOSE_2D.
runTranspose('2D, default axes [-2, -1]', Tensor::createFromData([
	[1, 2, 3],
	[4, 5, 6],
], 'matrix'));

// Fast path: TRANSPOSE_3D_LAST_TWO.
runTranspose('3D, default axes [-2, -1]', Tensor::createFromData([
	[[1, 2, 3], [4, 5, 6]],
	[[7, 8, 9], [10, 11, 12]],
], 'tokens'));

// Multi-head attention layout:
// [B, L, H, dk] -> [B, H, L, dk] -> [B, H, dk, L].
$attentionInput = Tensor::createFromData([
	[
		[[1, 2], [3, 4]],
		[[5, 6], [7, 8]],
		[[9, 10], [11, 12]],
	],
], 'attentionInput');
$attentionInput->setRequiresGrad(true);
$byHead = $attentionInput->transpose([1, 2]); // TRANSPOSE_4D_AXIS_1_2
$forScores = $byHead->transpose();             // TRANSPOSE_4D_LAST_TWO
$runtime = GraphRuntime::createFromOutputTensor($forScores);
$runtime->forward();
$runtime->backward();
$runtime->refreshTensorsData();

echo "\n=== Multi-head attention transpose ===\n";
echo '[B, L, H, dk]: [' . implode(', ', $attentionInput->shape) . "]\n";
echo '[B, H, L, dk]: [' . implode(', ', $byHead->shape) . "]\n";
echo '[B, H, dk, L]: [' . implode(', ', $forScores->shape) . "]\n";
$forScores->printData();
$attentionInput->printGrad();

// Fallback: TRANSPOSE_GENERIC (swap batch and head axes).
runTranspose('Generic axes [0, 2]', Tensor::createFromData([
	[
		[[1, 2], [3, 4], [5, 6]],
		[[7, 8], [9, 10], [11, 12]],
	],
	[
		[[13, 14], [15, 16], [17, 18]],
		[[19, 20], [21, 22], [23, 24]],
	],
], 'generic'), [0, 2]);
