param(
    [Parameter(Mandatory = $true)]
    [string]$LogPath
)

$Host.UI.RawUI.WindowTitle = 'jarvis - task activity'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Start at the CURRENT end of the log: a freshly spawned console shows
# nothing old, only new activity.
$offset = 0
if (Test-Path -LiteralPath $LogPath) {
    $offset = (Get-Item -LiteralPath $LogPath).Length
}

while ($true) {
    try {
        $length = (Get-Item -LiteralPath $LogPath).Length
        if ($length -lt $offset) {
            # A new task truncated the log - show it from the start.
            $offset = 0
            Clear-Host
        }
        if ($length -gt $offset) {
            $stream = [System.IO.File]::Open($LogPath, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
            try {
                $stream.Seek($offset, [System.IO.SeekOrigin]::Begin) | Out-Null
                $reader = New-Object System.IO.StreamReader($stream, [System.Text.Encoding]::UTF8)
                try {
                    $text = $reader.ReadToEnd()
                    if ($text.Length -gt 0) {
                        [Console]::Write($text)
                    }
                } finally {
                    $reader.Dispose()
                }
            } finally {
                $stream.Dispose()
            }
            $offset = $length
        }
    } catch {
        # Transient IO error (file mid-rewrite): ignore and retry.
    }
    Start-Sleep -Milliseconds 400
}
