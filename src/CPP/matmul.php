<?php

use PHP2xAI\Tensor\Tensor;
use PHP2xAI\Runtime\CPP\GraphRuntimeCpp;

include("../../vendor/autoload.php");

$a = Tensor::createFromData([
	[[1,2,3],[2,3,3]],
	[[3,4,1],[5,6,1]],
	[[3,4,2],[5,6,2]]
],"a");

$a->printData();

$b = Tensor::createFromData([
	[[1,2],[2,3],[3,4]],
	[[3,4],[5,6],[3,4]],
	[[3,4],[5,6],[2,1]]
], "b");

$b->print();

$c = $a->matmul($b);

// print_r($c->shape);

$graphRuntime = GraphRuntimeCpp::createFromOutputTensor($c);
$graphRuntime->forward();
$graphRuntime->backward();
$graphRuntime->refreshTensorsData(); // reload tensors data and grad after forward and backward

$c->printData();
$b->printGrad();
$a->printGrad();