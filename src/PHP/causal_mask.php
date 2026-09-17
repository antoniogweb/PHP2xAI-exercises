<?php

use PHP2xAI\Runtime\PHP\Core\GraphRuntime;
use PHP2xAI\Tensor\Tensor;

include("../../vendor/autoload.php");

function assertValues(string $label, array $actual, array $expected): void
{
	if (count($actual) !== count($expected))
		throw new RuntimeException("{$label}: size mismatch");

	foreach ($expected as $index => $value)
	{
		if (is_infinite($value))
		{
			if (!is_infinite($actual[$index]) || $actual[$index] > 0.0)
				throw new RuntimeException("{$label}: expected -INF at index {$index}");
		}
		else if (abs($actual[$index] - $value) > 1e-6)
		{
			throw new RuntimeException("{$label}: mismatch at index {$index}");
		}
	}
}

$Lq = 3;
$Lkv = 5;
$input = Tensor::createFromData([
	[
		[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
		[[16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]],
	],
], "scores");
$input->setRequiresGrad(true);
$output = $input->applyCausalMask($Lq, $Lkv);

$runtime = GraphRuntime::createFromOutputTensor($output);
$runtime->forward();
$runtime->backward();
$runtime->refreshTensorsData();

$expectedOutput = [
	1, 2, 3, -INF, -INF, 6, 7, 8, 9, -INF, 11, 12, 13, 14, 15,
	16, 17, 18, -INF, -INF, 21, 22, 23, 24, -INF, 26, 27, 28, 29, 30,
];
$expectedGrad = [
	1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
	1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
];

assertValues("forward", $output->data, $expectedOutput);
assertValues("backward", $input->grad, $expectedGrad);

echo "PHP causal-mask forward/backward: OK\n";
