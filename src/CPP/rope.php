<?php

use PHP2xAI\Runtime\CPP\GraphRuntimeCpp;
use PHP2xAI\Tensor\Tensor;

require __DIR__ . '/../../vendor/autoload.php';

function runRope(string $title, array $data, int $positionAxis, int $rotationAxis, int $offset, string $pairing): void
{
	$input = Tensor::createFromData($data, 'input');
	$input->setRequiresGrad(true);
	$output = $input->rope($positionAxis, $rotationAxis, $offset, 10000.0, $pairing);

	$context = $output->getContext();
	$inputId = $context->getTensorId($input);
	$outputId = $context->getTensorId($output);
	$graph = $context->export();
	$graph['loss'] = $outputId; // Makes dL/dY equal to one for the backward check.

	$runtime = new GraphRuntimeCpp($graph);
	$runtime->forward();
	$runtime->backward();

	echo "\n=== {$title} ===\n";
	echo 'kernel: ' . $graph['ops'][0]['attributes']['kernel'] . "\n";
	echo 'output: ' . implode(', ', $runtime->getTensorData($outputId)) . "\n";
	echo 'input gradient: ' . implode(', ', $runtime->getTensorGrad($inputId)) . "\n";
}

// Fast C++ kernel: [B, L, D], with position=L and rotation=D.
runRope(
	'LAST_TWO / INTERLEAVED',
	[
		[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
		[[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]],
	],
	-2,
	-1,
	3,
	Tensor::INTERLEAVED,
);

// Generic C++ kernel: position is axis 0, not the penultimate axis.
runRope(
	'GENERIC / ROTATE_HALF',
	[
		[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
		[[13.0, 14.0, 15.0, 16.0], [17.0, 18.0, 19.0, 20.0], [21.0, 22.0, 23.0, 24.0]],
	],
	0,
	2,
	1,
	Tensor::ROTATE_HALF,
);

echo "\nCPP RoPE forward/backward: OK\n";
