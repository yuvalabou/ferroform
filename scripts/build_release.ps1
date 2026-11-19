$ErrorActionPreference = "Stop"

Write-Host "Running Rust workspace tests..."
cargo test --workspace

Write-Host "Building release binaries..."
cargo build --release --workspace

$version = (Select-String -Path Cargo.toml -Pattern 'version\s*=\s*"([^"]+)"' | Select-Object -First 1).Matches.Groups[1].Value
if (-not $version) { $version = "2.0.0" }

$distDir = "dist"
New-Item -ItemType Directory -Force -Path $distDir | Out-Null

$stageDir = Join-Path $distDir "ferroform-$version-win64"
if (Test-Path $stageDir) { Remove-Item -Recurse -Force $stageDir }
New-Item -ItemType Directory -Force -Path $stageDir | Out-Null

Copy-Item "target\release\ferroform-gui.exe" $stageDir
Copy-Item "target\release\ferroform-cli.exe" $stageDir
Copy-Item "README.md" $stageDir -ErrorAction SilentlyContinue
Copy-Item "LICENSE" $stageDir -ErrorAction SilentlyContinue

$zipPath = Join-Path $distDir "ferroform-$version-win64.zip"
if (Test-Path $zipPath) { Remove-Item -Force $zipPath }
Compress-Archive -Path "$stageDir\*" -DestinationPath $zipPath

$hash = (Get-FileHash -Algorithm SHA256 $zipPath).Hash.ToLower()
"$hash  $(Split-Path -Leaf $zipPath)" | Out-File -Encoding ascii "$zipPath.sha256"

Write-Host "Release package created: $zipPath (SHA-256: $hash)"
