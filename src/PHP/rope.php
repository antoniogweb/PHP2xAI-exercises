<?php

use PHP2xAI\Runtime\CPP\GraphRuntimeCpp;
use PHP2xAI\Runtime\PHP\Core\GraphRuntime;
use PHP2xAI\Tensor\Tensor;

require __DIR__ . '/../../vendor/autoload.php';

const TOLERANCE = 1e-5;

function maxDifference(array $left, array $right): float
{
	if (count($left) !== count($right))
		throw new RuntimeException('Tensor sizes differ');

	$maximum = 0.0;
	foreach ($left as $index => $value)
		$maximum = max($maximum, abs($value - $right[$index]));

	return $maximum;
}

function runCase(string $name, array $data, int $positionAxis, int $rotationAxis, int $offset, string $pairing): array
{
	$input = Tensor::createFromData($data, 'input');
	$input->setRequiresGrad(true);
	$output = $input->rope($positionAxis, $rotationAxis, $offset, 10000.0, $pairing);

	$context = $output->getContext();
	$inputId = $context->getTensorId($input);
	$outputId = $context->getTensorId($output);
	$graph = $context->export();
	$graph['loss'] = $outputId; // dL/dY = 1: validates the backward rotation.

	$php = new GraphRuntime($graph);
	$php->forward();
	$php->backward();

	$cpp = new GraphRuntimeCpp($graph);
	$cpp->forward();
	$cpp->backward();

	$phpOutput = $php->getTensorData($outputId);
	$cppOutput = $cpp->getTensorData($outputId);
	$phpGrad = $php->getTensorGrad($inputId);
	$cppGrad = $cpp->getTensorGrad($inputId);
	$outputDifference = maxDifference($phpOutput, $cppOutput);
	$gradDifference = maxDifference($phpGrad, $cppGrad);

	if ($outputDifference > TOLERANCE || $gradDifference > TOLERANCE)
		throw new RuntimeException("{$name}: PHP/C++ mismatch");

	return [
		'name' => $name,
		'shape' => $input->shape,
		'position_axis' => $positionAxis,
		'rotation_axis' => $rotationAxis,
		'offset' => $offset,
		'base' => 10000.0,
		'pairing' => $pairing,
		'input' => $input->data,
		'php_output' => $phpOutput,
		'cpp_output' => $cppOutput,
		'php_grad' => $phpGrad,
		'cpp_grad' => $cppGrad,
		'php_cpp_output_max_difference' => $outputDifference,
		'php_cpp_grad_max_difference' => $gradDifference,
	];
}

$cases = [
	runCase('last-two / interleaved', [
		[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
		[[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]],
	], -2, -1, 3, Tensor::INTERLEAVED),
	runCase('generic / rotate-half', [
		[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
		[[13.0, 14.0, 15.0, 16.0], [17.0, 18.0, 19.0, 20.0], [21.0, 22.0, 23.0, 24.0]],
	], 0, 2, 1, Tensor::ROTATE_HALF),
];

if (in_array('--json', $argv, true))
{
	echo json_encode($cases, JSON_THROW_ON_ERROR), PHP_EOL;
	return;
}

foreach ($cases as $case)
{
	echo "{$case['name']}\n";
	echo 'PHP/C++ output max difference: ' . $case['php_cpp_output_max_difference'] . "\n";
	echo 'PHP/C++ gradient max difference: ' . $case['php_cpp_grad_max_difference'] . "\n\n";
}

echo "PHP and C++ RoPE forward/backward: OK\n";
echo "Run ../Torch/rope.py to compare both against PyTorch.\n";
