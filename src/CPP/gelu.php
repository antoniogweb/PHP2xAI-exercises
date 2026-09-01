<?php

use PHP2xAI\Runtime\CPP\GraphRuntimeCpp;
use PHP2xAI\Tensor\Tensor;

include("../../vendor/autoload.php");

function runGelu(string $name, Tensor $input): void
{
	$input->setRequiresGrad(true);
	$output = $input->gelu();
	$runtime = GraphRuntimeCpp::createFromOutputTensor($output);
	$runtime->forward();
	$runtime->backward();
	$runtime->refreshTensorsData();

	echo "\n=== {$name} ===\n";
	$output->printData();
	$input->printGrad();
}

// Values around zero and in both GELU tails.
runGelu('Matrix: negative, zero and positive values', Tensor::createFromData([
	[-3.0, -1.0, 0.0],
	[1.0, 2.0, 3.0],
], 'matrix'));

// GELU is elementwise and works unchanged on token embeddings [B, L, D].
runGelu('Token embeddings', Tensor::createFromData([
	[[-2.0, -0.5], [0.5, 2.0]],
	[[-1.5, 0.0], [1.5, 3.0]],
], 'tokens'));
