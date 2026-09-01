<?php

use PHP2xAI\Runtime\PHP\Core\GraphRuntime;
use PHP2xAI\Tensor\Tensor;

include("../../vendor/autoload.php");

function runScale(string $name, Tensor $input, float $scale): void
{
	$input->setRequiresGrad(true);
	$output = $input->scale($scale);
	$runtime = GraphRuntime::createFromOutputTensor($output);
	$runtime->forward();
	$runtime->backward();
	$runtime->refreshTensorsData();

	echo "\n=== {$name} ===\n";
	echo "scale: {$scale}\n";
	$output->printData();
	$input->printGrad();
}

runScale('Positive scalar on a matrix', Tensor::createFromData([
	[1.0, -2.0],
	[3.0, -4.0],
], 'matrix'), 2.5);

runScale('Negative scalar on token embeddings', Tensor::createFromData([
	[[1.0, 2.0], [3.0, 4.0]],
	[[5.0, 6.0], [7.0, 8.0]],
], 'tokens'), -0.5);

runScale('Zero scalar', Tensor::createFromData([1.0, -2.0, 3.0], 'vector'), 0.0);
