<?php
header('Content-Type: application/json');

$input = json_decode(file_get_contents('php://input'), true);
$k = isset($input['k']) ? intval($input['k']) : 2;
$points = isset($input['points']) ? $input['points'] : [];

function distance($a, $b) {
    return sqrt(pow($a['x'] - $b['x'], 2) + pow($a['y'] - $b['y'], 2));
}

function kmeans($points, $k, $max_iter = 100) {
    $n = count($points);
    if ($k > $n) $k = $n;

    shuffle($points);
    $centroids = array_slice($points, 0, $k);
    
    for ($iter = 0; $iter < $max_iter; $iter++) {
        $clusters = array_fill(0, $k, []);
        foreach ($points as $p) {
            $dist = [];
            foreach ($centroids as $c) $dist[] = distance($p, $c);
            $i = array_keys($dist, min($dist))[0];
            $clusters[$i][] = $p;
        }
        $newC = [];
        for ($i = 0; $i < $k; $i++) {
            if (count($clusters[$i]) === 0) {
                $newC[$i] = $points[array_rand($points)];
            } else {
                $sumX = $sumY = 0;
                foreach ($clusters[$i] as $pt) {
                    $sumX += $pt['x']; $sumY += $pt['y'];
                }
                $newC[$i] = ['x' => $sumX / count($clusters[$i]), 'y' => $sumY / count($clusters[$i])];
            }
        }
        if ($newC === $centroids) break;
        $centroids = $newC;
    }
    return $clusters;
}
$result = ['clusters' => kmeans($points, $k)];
echo json_encode($result);
?>