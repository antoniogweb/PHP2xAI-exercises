<?php

use PHP2xAI\Runtime\PHP\Core\GraphRuntime;
use PHP2xAI\Tensor\Tensor;

include("../../vendor/autoload.php");

function runReshape(string $name, Tensor $input, array $shape): void
{
	$input->setRequiresGrad(true);
	$output = $input->reshape($shape);
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

// [2, 3] -> [3, 2]: data order remains [1, 2, 3, 4, 5, 6].
runReshape('2D reshape', Tensor::createFromData([
	[1, 2, 3],
	[4, 5, 6],
], 'matrix'), [3, 2]);

// -1 infers the missing dimension: [2, 2, 3] -> [2, 6].
runReshape('Inferred dimension -1', Tensor::createFromData([
	[[1, 2, 3], [4, 5, 6]],
	[[7, 8, 9], [10, 11, 12]],
], 'tokens'), [2, -1]);

// Typical multi-head attention merge: [B, L, H, dk] -> [B, L, D].
runReshape('Merge attention heads', Tensor::createFromData([
	[
		[[1, 2], [3, 4]],
		[[5, 6], [7, 8]],
		[[9, 10], [11, 12]],
	],
], 'heads'), [1, 3, -1]);
