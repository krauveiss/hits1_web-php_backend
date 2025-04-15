<?php
class NeuralNetwork
{
    private $inputSize = 2500;
    private $hiddenSize = 256;
    private $outputSize = 10;
    private $biasHidden = array();
    private $biasOutput= array();
    private $weightsInputHidden;
    private $weightsHiddenOutput;

    public function __construct()
    {
        for ($i = 0; $i < $this->inputSize; $i++) {
            for ($j = 0; $j < $this->hiddenSize; $j++) {
                $this->weightsInputHidden[$i][$j] = rand(-50, 50) / 10000;
            }
        }

        for ($i = 0; $i < $this->hiddenSize; $i++) {
            for ($j = 0; $j < $this->outputSize; $j++) {
                $this->weightsHiddenOutput[$i][$j] = rand(-50, 50) / 10000;
            }
        }
        for ($i = 0; $i < $this->hiddenSize; $i++) {
            $this->biasHidden[$i] = rand(-500, 500) / 100; 
        }
        for ($i = 0; $i < $this->outputSize; $i++) {
            $this->biasOutput[$i] = rand(-500, 500) / 100;
        }

    }
    public function getWeights() {
        return [
            'input_hidden' => $this->weightsInputHidden,
            'hidden_output' => $this->weightsHiddenOutput,
            'bias_hidden' => $this->biasHidden,
            'bias_output' => $this->biasOutput
        ];
    }
    
    public function setWeights($weights) {
        $this->weightsInputHidden = $weights['input_hidden'];
        $this->weightsHiddenOutput = $weights['hidden_output'];
        $this->biasHidden = $weights['bias_hidden'];
        $this->biasOutput = $weights['bias_output'];
    }


    private function relu($x) {
        return max(0, $x);
    }

    private function softmax($arr) {
        $max = max($arr);
        $exp = array_map(function($x) use ($max) { 
            return exp($x - $max); 
        }, $arr);
        $sum = array_sum($exp);
        return array_map(function($x) use ($sum) { 
            return $x / $sum; 
        }, $exp);
    }

    public function predict($input)
    {
        $input = array_map('floatval', $input);
        $hiddenLayer = array();
        for ($i = 0; $i < $this->hiddenSize; $i++) {
            $sum = 0;
            for ($j = 0; $j < $this->inputSize; $j++) {
                $sum += $input[$j] * $this->weightsInputHidden[$j][$i];
            }
            $sum+=$this->biasHidden[$i];
            $hiddenLayer[$i]=$this->relu($sum);
        }

        $result=array();
        for ($i = 0; $i < $this->outputSize; $i++){
            $sum=0;
            for ($j = 0; $j < $this->hiddenSize; $j++){
                $sum += $hiddenLayer[$j]*$this-> weightsHiddenOutput[$j][$i]; 
            }
            $sum+=$this->biasOutput[$i];
            $result[$i]=$sum;
        }
        return $this->softmax($result);
    }

    public function loss($predicted, $actual) {
        $sum = 0;
        for ($i = 0; $i < $this->outputSize; $i++) {
            $sum += $actual[$i] * log($predicted[$i] + 1e-9);
        }
        return -$sum;
    }

    public function train($input,$target){
        $hiddenLayer = array();
        for ($i = 0; $i < $this->hiddenSize; $i++) {
            $sum = 0;
            for ($j = 0; $j < $this->inputSize; $j++) {
                $sum += $input[$j] * $this->weightsInputHidden[$j][$i];
            }
            $sum+=$this->biasHidden[$i];
            $hiddenLayer[$i]=$this->relu($sum);
        }

        $result=array();
        for ($i = 0; $i < $this->outputSize; $i++){
            $sum=0;
            for ($j = 0; $j < $this->hiddenSize; $j++){
                $sum += $hiddenLayer[$j]*$this-> weightsHiddenOutput[$j][$i]; 
            }
            $sum+=$this->biasOutput[$i];
            $result[$i]=$sum;
        }


        $grad_output = array();
        for($i = 0; $i < $this->outputSize; $i++){
            $grad_output[$i] = $this->softmax($result)[$i] - $target[$i];
        }
        
        $grad_hidden = array();
        for ($i=0; $i < $this->hiddenSize;$i++){
            $sum=0;
            for($j= 0; $j < $this->outputSize; $j++){
                $sum += $grad_output[$j] * $this->weightsHiddenOutput[$i][$j];
            }
            $grad_hidden[$i]=$sum * $hiddenLayer[$i] * (1-$hiddenLayer[$i]);
        }

        $learningRate = 0.01;

        for ($i = 0; $i < $this->hiddenSize; $i ++){
            for ($j = 0; $j < $this -> outputSize; $j++){
                $this->weightsHiddenOutput[$i][$j] -=  $learningRate * $grad_output[$j] * $hiddenLayer[$i];
            }
            
        }
        for ($i = 0; $i < $this->inputSize; $i++){
            for($j = 0; $j < $this->hiddenSize; $j++){
                $this->weightsInputHidden[$i][$j] -= $learningRate * $grad_hidden[$j] * $input[$i];
            }
        }

        for ($i= 0; $i < $this->outputSize; $i++){
            $this->biasOutput[$i] -= $learningRate * $grad_output[$i];
        }
        for ($i= 0; $i < $this->hiddenSize; $i++){
            $this->biasHidden[$i] -= $learningRate * $grad_hidden[$i];
        }
    }
        
    
}




$nn = new NeuralNetwork();
$nn->setWeights(json_decode(file_get_contents('train/weights50.json'),true));
$start_time = microtime(true);

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $gridData = isset($_POST['gridData']) ? $_POST['gridData'] : '';
    
    $input = json_decode($gridData, true);
    if ($input === null) {
        echo "Ошибка: неверные данные.";
        exit;
    }
    
    $output = $nn->predict($input);
    $predictedNum = array_search(max($output),$output);
    echo "<h2>Final predict:  " . $predictedNum . " <h2>";
    echo "<h1>/</h1>";
    foreach ($output as $key => $value) {
        echo "[$key] => " . number_format($value, 10) . "<br/>";
    }
} else {
    echo "Нет данных для обработки.";
}

$end_time = microtime(true);

$execution_time = $end_time - $start_time;


echo "Время выполнения: " . $execution_time . " секунд.";

