<?php
class NeuralNetwork
{
    private $inputSize = 25;
    private $hiddenSize = 15;
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


    private function sigmoid($x)
    {
        return 1 / (1 + exp(-$x));
    }

    private function softmax($arr) {
        $exp = array_map('exp', $arr);
        $sum = array_sum($exp);
        return array_map(function($x) use ($sum) { return $x / $sum; }, $exp);
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
            $hiddenLayer[$i]=$this->sigmoid($sum);
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
            $hiddenLayer[$i]=$this->sigmoid($sum);
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
// $one_hot = [0,1,0,0,0,0,0,0,0,0];
// $train_mat=[
//     0,0,1,0,0, 
//     0,1,1,0,0, 
//     0,0,1,0,0, 
//     0,0,1,0,0, 
//     0,0,1,0,0
// ];

$dataset = [
    [
        'input' => [
            0,1,1,1,0,
            1,0,0,0,1,
            1,0,0,0,1,
            1,0,0,0,1,
            0,1,1,1,0
        ],
        'target' => [1,0,0,0,0,0,0,0,0,0]
    ],

    [
        'input' => [
            0,0,0,0,0, 
            0,1,1,1,0, 
            0,1,0,1,0, 
            0,1,1,1,0, 
            0,0,0,0,0
        ],
        'target' => [1,0,0,0,0,0,0,0,0,0]
    ],

    
    [
        'input' => [
            0,0,1,0,0,
            0,1,1,0,0,
            0,0,1,0,0,
            0,0,1,0,0,
            0,1,1,1,0
        ],
        'target' => [0,1,0,0,0,0,0,0,0,0]
    ],
    
    [
        'input' => [
            0,1,1,1,0,
            0,0,0,1,0,
            0,1,1,1,0,
            0,1,0,0,0,
            0,1,1,1,0
        ],
        'target' => [0,0,1,0,0,0,0,0,0,0]
    ],


    [
        'input' => [
            1,1,1,1,0, 
            0,0,0,1,0, 
            0,1,1,0,0, 
            1,1,1,1,0, 
            0,0,0,0,0
        ],
        'target' => [0,0,1,0,0,0,0,0,0,0]
    ],
    
    [
        'input' => [
            0,1,1,1,0,
            0,0,0,1,0,
            0,1,1,1,0,
            0,0,0,1,0,
            0,1,1,1,0
        ],
        'target' => [0,0,0,1,0,0,0,0,0,0]
    ],

    [
        'input' => [
            1,1,1,1,0, 
            0,0,0,1,0, 
            1,1,1,1,0, 
            0,0,0,1,0, 
            0,1,1,1,0
        ],
        'target' => [0,0,0,1,0,0,0,0,0,0]
    ],

    
    [
        'input' => [
            1,0,0,1,0,
            1,0,0,1,0,
            1,1,1,1,0,
            0,0,0,1,0,
            0,0,0,1,0
        ],
        'target' => [0,0,0,0,1,0,0,0,0,0]
    ],
    
    [
        'input' => [
            1,1,1,1,0,
            1,0,0,0,0,
            1,1,1,1,0,
            0,0,0,1,0,
            1,1,1,1,0
        ],
        'target' => [0,0,0,0,0,1,0,0,0,0]
    ],
    
    [
        'input' => [
            0,1,1,1,0,
            1,0,0,0,0,
            1,1,1,1,0,
            1,0,0,1,0,
            0,1,1,1,0
        ],
        'target' => [0,0,0,0,0,0,1,0,0,0]
    ],

    [
        'input' => [
            1,1,1,0,0, 
            1,0,0,0,0, 
            1,1,1,0,0, 
            1,0,1,0,0, 
            1,1,1,0,0
        ],
        'target' => [0,0,0,0,0,0,1,0,0,0]
    ],
    
    

    [
        'input' => [
            1,1,1,1,0,
            0,0,0,1,0,
            0,0,1,0,0,
            0,1,0,0,0,
            1,0,0,0,0
        ],
        'target' => [0,0,0,0,0,0,0,1,0,0]
    ],
    

    [
        'input' => [
            0,1,1,1,0,
            1,0,0,1,0,
            0,1,1,1,0,
            1,0,0,1,0,
            0,1,1,1,0
        ],
        'target' => [0,0,0,0,0,0,0,0,1,0]
    ],

    [
        'input' => [
            0,1,1,1,0,
            1,0,0,1,0,
            0,1,1,1,0,
            0,0,0,1,0,
            0,1,1,1,0
        ],
        'target' => [0,0,0,0,0,0,0,0,0,1]
    ]
];

$epochs = 10000;
for ($epoch = 0; $epoch < $epochs; $epoch++) {
    foreach ($dataset as $data) {
        $nn->train($data['input'], $data['target']);
    }
}
$trainedWeights = $nn->getWeights();
file_put_contents('train/weights.json', json_encode($trainedWeights));