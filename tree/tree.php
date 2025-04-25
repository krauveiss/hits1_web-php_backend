<?php

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
        if (!isset($counts[$label])) $counts[$label] = 0;
        $counts[$label]++;
    }
    $entropy = 0.0;
    $total = count($rows);
    foreach ($counts as $count) {
        $p = $count / $total;
        $entropy -= $p * log($p, 2);
    }
    return $entropy;
}


function information_gain($rows, $col_index) {
    $base_entropy = entropy($rows);
    $subsets = [];
    foreach ($rows as $row) {
        $key = $row[$col_index];
        $subsets[$key][] = $row;
    }
    $weighted_entropy = 0.0;
    $total = count($rows);
    foreach ($subsets as $subset) {
        $weighted_entropy += (count($subset) / $total) * entropy($subset);
    }
    return $base_entropy - $weighted_entropy;
}


function build_tree($rows, $col_names) {
    if (empty($rows)) {
        return ['type'=>'leaf', 'label'=>null];
    }

    $labels = array_column($rows, count($rows[0]) - 1);
    $unique_labels = array_unique($labels);
    if (count($unique_labels) === 1) {
        return ['type'=>'leaf', 'label'=>$unique_labels[0]];
    }

    if (count($col_names) <= 1) {
        $counts = array_count_values($labels);
        arsort($counts);
        return ['type'=>'leaf', 'label'=>array_key_first($counts)];
    }
    
    $best_gain = -INF;
    $best_index = null;
    foreach ($col_names as $i => $col) {
        if ($i === count($col_names) - 1) continue;
        $gain = information_gain($rows, $i);
        if ($gain > $best_gain) {
            $best_gain = $gain;
            $best_index = $i;
        }
    }
    $best_attr = $col_names[$best_index];
    $tree = ['type'=>'node', 'attribute'=>$best_attr, 'index'=>$best_index, 'branches'=>[]];
    $subsets = [];
    foreach ($rows as $row) {
        $subsets[$row[$best_index]][] = $row;
    }
    foreach ($subsets as $value => $subset) {
        $new_cols = $col_names;
        unset($new_cols[$best_index]);
        $new_cols = array_values($new_cols);
        $new_rows = [];
        foreach ($subset as $r) {
            $new_rows[] = array_values(array_diff_key($r, [$best_index=>0]));
        }
        $tree['branches'][$value] = build_tree($new_rows, $new_cols);
    }
    return $tree;
}


function print_tree($tree, $indent = "") {
    if ($tree['type'] === 'leaf') {
        echo $indent . "Leaf: " . ($tree['label'] ?? 'null') . "<br>";
        return;
    }
    echo $indent . "[" . htmlspecialchars($tree['attribute']) . "]<br>";
    foreach ($tree['branches'] as $val => $branch) {
        echo $indent . "&nbsp;&nbsp;-- $val --> ";
        print_tree($branch, $indent . '&nbsp;&nbsp;&nbsp;&nbsp;');
    }
}


function classify_with_path($tree, $instance, $col_names, &$path = []) {
    if ($tree['type'] === 'leaf') {
        $path[] = "Reached leaf: " . ($tree['label'] ?? 'null');
        return $tree['label'];
    }
    $attr = $tree['attribute'];
    $index = array_search($attr, $col_names);
    $value = $instance[$index] ?? null;
    $path[] = "Node {$attr} = {$value}";
    if (!isset($tree['branches'][$value])) {
        $branch = reset($tree['branches']);
    } else {
        $branch = $tree['branches'][$value];
    }
    $new_instance = array_values(array_diff_key($instance, [$index=>0]));
    $new_cols = $col_names;
    unset($new_cols[$index]);
    $new_cols = array_values($new_cols);
    return classify_with_path($branch, $new_instance, $new_cols, $path);
}


$train_data = $_POST['train_data'] ?? '';
$test_data = $_POST['test_data'] ?? '';
$action = $_POST['action'] ?? '';

$columns = [];
$tree = null;
$result = '';

$tree_print = '';

if ($action === 'build' && trim($train_data) !== '') {
    $all = parse_csv_text($train_data);
    $columns = array_shift($all);
    $tree = build_tree($all, $columns);
    ob_start(); print_tree($tree); $tree_print = ob_get_clean();
}

if ($action === 'classify' && trim($train_data) !== '' && trim($test_data) !== '') {
    $all = parse_csv_text($train_data);
    $columns = array_shift($all);
    $tree = build_tree($all, $columns);

    ob_start(); print_tree($tree); $tree_print = ob_get_clean();
    $tests = parse_csv_text($test_data);
    foreach ($tests as $t) {
        $path = [];
        $label = classify_with_path($tree, $t, $columns, $path);
        $result .= '<b>Example:</b> ' . implode(',', $t) . '<br>';
        $result .= '<b>Path:</b><br>' . implode('<br>', $path) . '<br>';
        $result .= '<b>Prediction: </b>' . $label . '<hr>';
    }
}
?>



<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Дерево решений ID3</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;700;800&display=swap');
    * { margin:0; padding:0; box-sizing:border-box; }
    body { background: radial-gradient(circle, rgba(15,24,36,1) 0%, rgba(5,5,5,1) 58%); font-family:'Poppins',sans-serif; color:white; }
    header { text-align:center; padding:20px 0; }
    header p { font-size:30px; font-weight:600; background: linear-gradient(90deg, #fff 0%, #3BA0FF 63%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .container { width:90%; max-width:800px; margin:40px auto; display:flex; flex-direction:column; gap:30px; }
    .panel { background: rgba(10,10,10,0.8); border-radius:10px; padding:20px; }
    .panel h2 { margin-bottom:20px; font-weight:500; }
    textarea { width:100%; height:120px; background:#0a0a0a; border:1px solid #404040; border-radius:5px; color:white; padding:10px; font-family:monospace; resize: vertical; }
    .btn-main { border:none; background-color:#3BA0FF; color:white; border-radius:5px; width:160px; height:45px; cursor:pointer; transition:all 0.4s; font-size:16px; font-family:'Poppins',sans-serif; }
    .btn-main:hover { color:#202020; font-size:17px; box-shadow:0 0 10px #0F1824FF; }
    .btn-secondary { border:1px solid #a1a1a1; background:transparent; color:white; border-radius:5px; width:160px; height:45px; cursor:pointer; transition:all 0.4s; font-size:16px; font-family:'Poppins',sans-serif; }
    .btn-secondary:hover { font-size:17px; text-shadow:0 0 10px white; box-shadow:0 0 3px white; }
    .form-group { display:flex; flex-direction:column; gap:10px; margin-bottom:20px; }
    .output { background: rgba(20,20,20,0.9); border:1px solid #404040; border-radius:5px; padding:15px; font-family:monospace; font-size:14px; color:#eef; max-height:300px; overflow:auto; }
  </style>
</head>
<body>
  <header><p>Дерево решений ID3</p></header>
  <div class="container">
    <form method="post" class="panel">
      <h2>Обучающая выборка (CSV ";")</h2>
      <div class="form-group">
        <textarea name="train_data"><?php echo htmlspecialchars($train_data); ?></textarea>
      </div>
      <div style="display:flex; gap:20px;">
        <button type="submit" name="action" value="build" class="btn-main">Построить дерево</button>
        <button type="submit" name="action" value="classify" class="btn-secondary">Классифицировать</button>
      </div>
    </form>

    <?php if ($tree_print): ?>
    <div class="panel">
      <h2>Структура дерева</h2>
      <div class="output"><?php echo $tree_print; ?></div>
    </div>
    <?php endif; ?>

    <?php if ($action === 'classify'): ?>
    <form method="post" class="panel">
      <h2>Новые примеры</h2>
      <input type="hidden" name="train_data" value="<?php echo htmlspecialchars($train_data); ?>">
      <div class="form-group">
        <textarea name="test_data"><?php echo htmlspecialchars($test_data); ?></textarea>
      </div>
      <button type="submit" name="action" value="classify" class="btn-main">Классифицировать</button>
    </form>

    <?php if ($result): ?>
      <div class="panel">
        <h2>Результаты классификации</h2>
        <div class="output"><?php echo $result; ?></div>
      </div>
    <?php endif; ?>
    <?php endif; ?>
  </div>
</body>
</html>

