<?php

use PHP2xAI\Tensor\Tensor;
use PHP2xAI\Runtime\PHP\Core\GraphRuntime;

include("../../vendor/autoload.php");

$a = Tensor::createFromData([
							[[1,2],[2,3]],
							[[1,4],[1,1]]
							 ],"a");

$a->printData();

$b = $a->softmax(1);

// print_r($b->context->export());

$graphRuntime = GraphRuntime::createFromOutputTensor($b);
$graphRuntime->forward();
$graphRuntime->backward();
$graphRuntime->refreshTensorsData(); // reload tensors data and grad after forward and backward

$b->printData();
$a->printGrad();