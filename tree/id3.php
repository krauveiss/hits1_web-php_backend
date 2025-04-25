<?php
header('Content-Type: application/json');

function parse_csv_text($text) {
    $lines = array_filter(array_map('trim', explode("\n", trim($text))));
    $data = [];
    foreach ($lines as $line) {
        $data[] = str_getcsv($line, ';');
    }
    return $data;
}

function entropy($rows) {
    $counts = [];
    foreach ($rows as $row) {
        $label = end($row);
        $counts[$label] = ($counts[$label] ?? 0) + 1;
    }
    $H = 0.0;
    $n = count($rows);
    foreach ($counts as $c) {
        $p = $c / $n;
        $H -= $p * log($p, 2);
    }
    return $H;
}

function information_gain($rows, $col) {
    $base = entropy($rows);
    $subs = [];
    foreach ($rows as $r) {
        $subs[$r[$col]][] = $r;
    }
    $we = 0.0;
    $n = count($rows);
    foreach ($subs as $s) {
        $we += (count($s)/$n) * entropy($s);
    }
    return $base - $we;
}

function build_tree($rows, $cols) {
    if (empty($rows)) return ['type'=>'leaf','label'=>null];
    $labels = array_column($rows, count($rows[0]) - 1);
    if (count(array_unique($labels)) === 1) {
        return ['type'=>'leaf','label'=>$labels[0]];
    }
    if (count($cols) <= 1) {
        $cnt = array_count_values($labels);
        arsort($cnt);
        return ['type'=>'leaf','label'=>array_key_first($cnt)];
    }
    $bestGain = -INF; $bestIdx = null;
    foreach ($cols as $i => $c) {
        if ($i === count($cols)-1) continue;
        $g = information_gain($rows, $i);
        if ($g > $bestGain) { $bestGain = $g; $bestIdx = $i; }
    }
    $attr = $cols[$bestIdx];
    $tree = ['type'=>'node','attribute'=>$attr,'branches'=>[]];
    $subs = [];
    foreach ($rows as $r) {
        $subs[$r[$bestIdx]][] = $r;
    }
    foreach ($subs as $val => $subset) {
        $newCols = array_values(array_diff_key($cols, [$bestIdx=>0]));
        $newRows = [];
        foreach ($subset as $r) {
            $newRows[] = array_values(array_diff_key($r, [$bestIdx=>0]));
        }
        $tree['branches'][$val] = build_tree($newRows, $newCols);
    }
    return $tree;
}

function render_tree_html($t) {
    if ($t['type'] === 'leaf') {
        return '<li><em>Leaf:</em> ' . htmlspecialchars($t['label'] ?? 'null') . '</li>';
    }
    $html = '<li><strong>' . htmlspecialchars($t['attribute']) . '</strong><ul>';
    foreach ($t['branches'] as $val => $b) {
        $html .= '<li>' . htmlspecialchars($val) . '<ul>' . render_tree_html($b) . '</ul></li>';
    }
    $html .= '</ul></li>';
    return $html;
}

function classify_with_path($t, $inst, $cols, &$path = []) {
    if ($t['type'] === 'leaf') {
        $path[] = 'Leaf: ' . ($t['label'] ?? 'null');
        return $t['label'];
    }
    $i = array_search($t['attribute'], $cols);
    $val = $inst[$i] ?? null;
    $path[] = $t['attribute'] . ' = ' . $val;
    $branch = $t['branches'][$val] ?? reset($t['branches']);
    $newInst = array_values(array_diff_key($inst, [$i=>0]));
    $newCols = array_values(array_diff_key($cols, [$i=>0]));
    return classify_with_path($branch, $newInst, $newCols, $path);
}

$train = $_POST['train_data'] ?? '';
$test  = $_POST['test_data']  ?? '';
$action= $_POST['action']     ?? '';

$data = parse_csv_text($train);
$cols = array_shift($data);
$tree = build_tree($data, $cols);
$tree_html = '<ul>' . render_tree_html($tree) . '</ul>';

$result_html = '';
if ($action === 'classify' && trim($test) !== '') {
    $tests = parse_csv_text($test);
    foreach ($tests as $t) {
        $path=[];
        $label = classify_with_path($tree, $t, $cols, $path);
        $result_html .= '<div><strong>Sample:</strong> ' . htmlspecialchars(implode(',', $t)) . '</div>';
        $result_html .= '<div><strong>Direction:</strong> ' . htmlspecialchars(implode(' → ', $path)) . '</div>';
        $result_html .= '<div><strong>Result:</strong> ' . htmlspecialchars($label) . '</div><hr>';
    }
}

echo json_encode(['tree' => $tree_html, 'result' => $result_html]);