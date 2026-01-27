<?php

use PHP2xAI\Tensor\Tensor;
use PHP2xAI\Runtime\PHP\Core\GraphRuntime;

include("../../vendor/autoload.php");

$a = Tensor::createFromData([[1,2],[2,3]],"a");

echo $a->get([1,1])."\n";

$a->set([1,1], 6);

echo $a->get([1,1])."\n";

$a->printData();

$b = Tensor::createFromData([[1,2],[2,3]], "b");

$b->print();

$c = $a->matmul($b);

$graphRuntime = GraphRuntime::createFromOutputTensor($c);
$graphRuntime->forward();
$graphRuntime->backward();
$graphRuntime->refreshTensorsData(); // reload tensors data and grad after forward and backward

$c->printData();
$b->printGrad();
$a->printGrad();

// print_r($context);
