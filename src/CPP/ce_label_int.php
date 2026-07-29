<?php

use PHP2xAI\Tensor\Tensor;
use PHP2xAI\Runtime\CPP\GraphRuntimeCpp;

include("../../vendor/autoload.php");

function runCeLogitsLabelInt(array $logitsData, array $targetData, int $axis, string $title): void
{
	$y = Tensor::createFromData($logitsData, "y");
	$target = Tensor::createFromData($targetData, "target");

	echo "\n=== {$title} ===\n";
	echo "Logits:\n";
	$y->printData();

	echo "Target label int:\n";
	$target->printData();

	$c = $y->CELogitsLabelInt($target, $axis);

	$graphRuntime = GraphRuntimeCpp::createFromOutputTensor($c);
	$graphRuntime->forward();
	$graphRuntime->backward();
	$graphRuntime->refreshTensorsData();

	echo "Loss:\n";
	$c->printData();

	echo "Target grad:\n";
	$target->printGrad();

	echo "Logits grad:\n";
	$y->printGrad();
}

// Fast path: logits shape [batch, steps, classes], CE along the last axis.
runCeLogitsLabelInt(
	[
		[[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]],
		[[-1.0, 0.0, 1.0], [3.0, 1.0, -2.0]],
	],
	[
		[2, 1],
		[0, 0],
	],
	-1,
	"3D last axis"
);

// Generic axis: logits shape [batch, classes, features], CE along axis 1.
runCeLogitsLabelInt(
	[
		[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
		[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
	],
	[
		[2, 0],
		[1, 2],
	],
	1,
	"3D generic axis 1"
);
