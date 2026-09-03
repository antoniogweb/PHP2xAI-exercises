<?php

use PHP2xAI\Runtime\PHP\Core\GraphRuntime;
use PHP2xAI\Tensor\Tensor;

include("../../vendor/autoload.php");

function runLayerNorm(string $name, Tensor $input, Tensor $gamma, Tensor $beta, int $axis = -1): void
{
	$input->setRequiresGrad(true);
	$gamma->setRequiresGrad(true);
	$beta->setRequiresGrad(true);

	$output = $input->layerNorm($gamma, $beta, $axis);
	$runtime = GraphRuntime::createFromOutputTensor($output);
	$runtime->forward();
	$runtime->backward();
	$runtime->refreshTensorsData();

	echo "\n=== {$name} ===\n";
	echo "output:\n";
	$output->printData();
	echo "input grad:\n";
	$input->printGrad();
	echo "gamma grad:\n";
	$gamma->printGrad();
	echo "beta grad:\n";
	$beta->printGrad();
}

// Fast path: every token of [B, L, D] is normalized independently over D.
runLayerNorm(
	'Transformer tokens [B, L, D], last axis',
	Tensor::createFromData([
		[[1.0, -2.0, 3.5], [4.0, 0.5, -1.0]],
		[[2.0, 8.0, -3.0], [1.5, 2.5, 6.0]],
	], 'tokens'),
	Tensor::createFromData([1.2, -0.7, 0.4], 'gamma'),
	Tensor::createFromData([0.1, 0.2, -0.3], 'beta'),
);

// Generic path: normalize H, not the final D, for each [B, :, L, D] slice.
runLayerNorm(
	'Generic axis 1 on [B, H, L, D]',
	Tensor::createFromData([
		[
			[[1.0, 2.0], [3.0, 4.0]],
			[[5.0, 6.0], [7.0, 8.0]],
			[[2.0, -1.0], [0.5, 3.0]],
		],
		[
			[[-2.0, 1.0], [4.0, -3.0]],
			[[0.0, 2.0], [1.0, 5.0]],
			[[3.0, 4.0], [-1.0, 2.0]],
		],
	], 'attention'),
	Tensor::createFromData([1.0, 0.5, -0.25], 'gamma_h'),
	Tensor::createFromData([0.0, 0.1, -0.2], 'beta_h'),
	1,
);
