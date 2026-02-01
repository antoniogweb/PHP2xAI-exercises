<?php

use PHP2xAI\Tensor\Tensor;
use PHP2xAI\Runtime\PHP\Core\GraphRuntime;

include("../../vendor/autoload.php");

$a = Tensor::createFromData([[[1,2],[2,3]],[[3,4],[5,6]]],"a");

$a->printData();

$b = Tensor::createFromData([1,2], "b");

$b->print();

$c = $a->add($b);

// print_r($c->context->export());

$graphRuntime = GraphRuntime::createFromOutputTensor($c);
$graphRuntime->forward();
$graphRuntime->backward();
$graphRuntime->refreshTensorsData(); // reload tensors data and grad after forward and backward

$c->printData();
$b->printGrad();
$a->printGrad();