# Sequential training: text (resumes if a checkpoint exists), then image. Re-run safely after any interruption.
Set-Location $PSScriptRoot
$env:PYTHONUNBUFFERED = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

$textArgs = @("training/train_text_classifier.py", "--train-csv", "data/processed/text/hc3_train.csv",
  "--val-csv", "data/processed/text/hc3_val.csv", "--test-csv", "data/processed/text/hc3_test.csv",
  "--epochs", "20", "--patience", "3")
if (Test-Path "models/text_dl_last.pt") { $textArgs += "--resume" }
if (-not (Test-Path "eval/results/text_dl/test_metrics.json")) {
  python @textArgs
}

$imgArgs = @("training/train_image_model.py", "--data-dir", "data/processed/image/wilddeepfake", "--epochs", "20", "--patience", "4")
if (Test-Path "models/efficientnet_b4_last_checkpoint.pt") { $imgArgs += "--resume" }
python @imgArgs
Write-Host "ALL TRAINING DONE"
