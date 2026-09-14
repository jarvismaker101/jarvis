import os
import subprocess
import time
import webbrowser
import glob

import requests

from backend.core import deadline as budget


#: F24: the YouTube lookup used to have no timeout at all, so a stalled
#: request could outlive any caller's budget. It is now bounded, and an
#: expired budget sends no request.
YOUTUBE_LOOKUP_TIMEOUT = 10.0

#: Pause between the actions of one batch. F24: taken from the remaining
#: budget (never granted fresh time after it expired).
BATCH_PAUSE_SECONDS = 1.0


def get_first_youtube_video(query, deadline=None, timeout=YOUTUBE_LOOKUP_TIMEOUT):
    """Resolve a YouTube search to the first video URL (None if none/expired).

    *deadline* (F24) is the shared budget handle: when it is already spent
    no request is sent, and the request's own timeout is sliced to whatever
    is left of the window.
    """
    handle = budget.resolve(deadline)
    if handle is not None and handle.stopped():
        print("[EXEC] youtube lookup skipped — budget exhausted")
        return None
    request_timeout = budget.seconds_for(handle, timeout)
    if request_timeout is None:
        return None
    try:
        url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
        response = requests.get(
            url, headers={"User-Agent": "Mozilla/5.0"}, timeout=request_timeout
        )
        html = response.text

        start = html.find("/watch?v=")
        if start == -1:
            return None

        video_id = html[start : start + 20].split('"')[0]
        return "https://www.youtube.com" + video_id
    except Exception:
        return None


def open_in_browser(url, browser=None):
    """F47: route "open this URL" through the broker's explicit decision.

    The broker knows which browser sessions are live, so opening stops being
    a guess:

      * ``research_page`` — the warm research browser is running: open a page
        in IT (same profile, same authentication) instead of a cold default
        browser;
      * ``focus`` / ``attach`` — the URL is already open in a known session:
        bring THAT tab forward instead of opening a second copy;
      * ``external`` / un-honored decisions — the user's default browser,
        recorded via ``note_external_launch`` exactly as before.

    Best-effort throughout: any broker/transport problem falls back to the
    pre-G6 behavior (a managed launch), never an error to the caller.
    """
    try:
        from backend.services import browser_session_broker

        decision = browser_session_broker.route_open(url)
    except Exception:
        decision = {"action": "external"}

    action = decision.get("action")
    if action == "research_page" and _open_in_research_browser(url):
        return True
    if action in ("attach", "focus") and _focus_known_tab(decision, url):
        return True

    opened = _launch_in_browser(url, browser)
    if opened:
        # F47: even an unmanaged launch is browser-session identity — record
        # it so the broker can say "that URL is open in the user's browser"
        # instead of knowing nothing. Best-effort by design.
        try:
            from backend.services import browser_session_broker

            browser_session_broker.note_external_launch(url, browser)
        except Exception:
            pass
    return opened


def _open_in_research_browser(url):
    """Open *url* as a page of the warm research browser (same profile).

    Guarded by ``research_browser.is_warm()`` so a cold open can never START
    the worker as a side effect — reuse only, or fall back.
    """
    try:
        from backend.services import research_browser

        if not research_browser.is_warm():
            return False

        def _job(page):
            page.goto(url, timeout=60000, wait_until="domcontentloaded")
            return page.url

        research_browser.submit(_job, task_id="executor-open", timeout=70)
        return True
    except Exception as exc:
        print("[EXEC] research-browser open failed:", exc)
        return False


def _focus_known_tab(decision, url):
    """Bring an already-open tab forward instead of opening a second copy.

    Only the agent's daemon browser can be focused from here (via the MCP
    ``switch_tab``); other sessions' tabs are their owner's business. The
    daemon must already be listening — this never starts one. Best-effort:
    any failure falls back to the external launch.
    """
    session = decision.get("session") or {}
    if str(session.get("owner") or "") != "agent":
        return False
    try:
        from backend.services import opencode_client
        from backend.services.brave_mcp_client import BraveMcpClient

        if not opencode_client._pids_on_port(opencode_client.BRAVE_MCP_PORT):
            return False
        client = BraveMcpClient(timeout=8)
        try:
            client.connect()
            client.call_tool("switch_tab", {"url_contains": url})
        finally:
            client.close()
        return True
    except Exception:
        return False


def _launch_in_browser(url, browser=None):
    try:
        if browser == "chrome":
            chrome_path = os.getenv(
                "JARVIS_CHROME_PATH",
                "C:/Program Files/Google/Chrome/Application/chrome.exe",
            )
            if os.path.exists(chrome_path):
                subprocess.Popen([chrome_path, url])
                return True

        elif browser == "edge":
            edge_path = os.getenv(
                "JARVIS_EDGE_PATH",
                "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
            )
            if os.path.exists(edge_path):
                subprocess.Popen([edge_path, url])
                return True

        elif browser == "brave":
            brave_path = os.getenv(
                "JARVIS_BRAVE_PATH",
                os.path.join(
                    os.environ.get("LOCALAPPDATA", ""),
                    "BraveSoftware",
                    "Brave-Browser",
                    "Application",
                    "brave.exe",
                ),
            )
            if os.path.exists(brave_path):
                subprocess.Popen([brave_path, url])
                return True

        return webbrowser.open(url)
    except Exception as exc:
        print("[EXEC] Browser fallback:", exc)
        return webbrowser.open(url)


def launch_app(app_name):
    """Attempt to launch a local application. Return True if successful, else False."""
    name = app_name.lower().strip()

    # 1. Common system utilities
    system_apps = {
        "notepad": "notepad.exe",
        "calc": "calc.exe",
        "calculator": "calc.exe",
        "paint": "mspaint.exe",
        "mspaint": "mspaint.exe",
        "cmd": "cmd.exe",
        "command prompt": "cmd.exe",
        "explorer": "explorer.exe",
        "file explorer": "explorer.exe",
        "taskmgr": "taskmgr.exe",
        "task manager": "taskmgr.exe",
        "write": "write.exe",
        "wordpad": "write.exe",
    }

    if name in system_apps:
        try:
            subprocess.Popen(system_apps[name])
            print(f"[EXEC] Launched system app: {system_apps[name]}")
            return True
        except Exception as e:
            print(f"[EXEC] Failed to launch system app {name}: {e}")

    # 2. Known apps with special locations/protocols
    # Brave
    if name == "brave":
        brave_path = os.getenv(
            "JARVIS_BRAVE_PATH",
            os.path.join(
                os.environ.get("LOCALAPPDATA", ""),
                "BraveSoftware",
                "Brave-Browser",
                "Application",
                "brave.exe",
            ),
        )
        if os.path.exists(brave_path):
            try:
                subprocess.Popen([brave_path])
                print(f"[EXEC] Launched Brave: {brave_path}")
                return True
            except Exception as e:
                print(f"[EXEC] Failed to launch Brave: {e}")
        # Try Program Files paths
        for root_path in (os.environ.get("ProgramFiles", ""), os.environ.get("ProgramFiles(x86)", "")):
            if root_path:
                alt_path = os.path.join(root_path, "BraveSoftware", "Brave-Browser", "Application", "brave.exe")
                if os.path.exists(alt_path):
                    try:
                        subprocess.Popen([alt_path])
                        print(f"[EXEC] Launched Brave (alt): {alt_path}")
                        return True
                    except Exception as e:
                        print(f"[EXEC] Failed to launch Brave (alt): {e}")

    # Chrome
    if name == "chrome":
        chrome_path = os.getenv(
            "JARVIS_CHROME_PATH",
            "C:/Program Files/Google/Chrome/Application/chrome.exe",
        )
        if os.path.exists(chrome_path):
            try:
                subprocess.Popen([chrome_path])
                print(f"[EXEC] Launched Chrome: {chrome_path}")
                return True
            except Exception as e:
                print(f"[EXEC] Failed to launch Chrome: {e}")

    # Edge
    if name == "edge":
        try:
            subprocess.Popen("start msedge", shell=True)
            print(f"[EXEC] Launched Edge via shell")
            return True
        except Exception as e:
            print(f"[EXEC] Failed to launch Edge: {e}")

    # WhatsApp
    if name == "whatsapp":
        try:
            os.startfile("whatsapp:")
            print(f"[EXEC] Launched Whatsapp protocol")
            return True
        except Exception:
            matches = sorted(
                glob.glob(
                    r"C:\Program Files\WindowsApps\5319275A.51895FA4EA97F_*__cv1g1gvanyjgm\WhatsApp.Root.exe"
                ),
                reverse=True,
            )
            if matches:
                try:
                    os.startfile(matches[0])
                    print(f"[EXEC] Launched WhatsApp exe: {matches[0]}")
                    return True
                except Exception as e:
                    print(f"[EXEC] Failed to launch WhatsApp exe: {e}")

    # Spotify
    if name == "spotify":
        try:
            os.startfile("spotify:")
            print(f"[EXEC] Launched Spotify protocol")
            return True
        except Exception as e:
            print(f"[EXEC] Failed to launch Spotify protocol: {e}")

    # VS Code / Code
    if name in ("vs code", "vscode", "code"):
        vscode_path = os.path.join(
            os.environ.get("LOCALAPPDATA", ""),
            "Programs",
            "Microsoft VS Code",
            "Code.exe"
        )
        if os.path.exists(vscode_path):
            try:
                subprocess.Popen([vscode_path])
                print(f"[EXEC] Launched VS Code: {vscode_path}")
                return True
            except Exception as e:
                print(f"[EXEC] Failed to launch VS Code: {e}")
        else:
            try:
                subprocess.Popen("code", shell=True)
                print(f"[EXEC] Launched VS Code via PATH")
                return True
            except Exception as e:
                print(f"[EXEC] Failed to launch VS Code via PATH: {e}")

    # Antigravity
    if name == "antigravity":
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        lnk = os.path.join(desktop_path, "Antigravity.lnk")
        if os.path.exists(lnk):
            try:
                os.startfile(lnk)
                print(f"[EXEC] Launched Antigravity shortcut: {lnk}")
                return True
            except Exception as e:
                print(f"[EXEC] Failed to launch Antigravity shortcut: {e}")

    # 3. Dynamic search in Desktop & Start Menu folders for any .lnk shortcut matching name
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    start_menu_paths = [
        os.path.join(os.environ.get("PROGRAMDATA", ""), "Microsoft", "Windows", "Start Menu", "Programs"),
        os.path.join(os.environ.get("APPDATA", ""), "Microsoft", "Windows", "Start Menu", "Programs"),
    ]

    desktops = [desktop_path]
    onedrive_desktop = os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop")
    if os.path.exists(onedrive_desktop):
        desktops.append(onedrive_desktop)

    for d_path in desktops:
        if os.path.exists(d_path):
            for lnk in glob.glob(os.path.join(d_path, "*.lnk")):
                if name in os.path.basename(lnk).lower():
                    try:
                        os.startfile(lnk)
                        print(f"[EXEC] Launched {lnk} from Desktop")
                        return True
                    except Exception as e:
                        print(f"[EXEC] Failed to launch {lnk}: {e}")

    for folder in start_menu_paths:
        if os.path.exists(folder):
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if file.lower().endswith(".lnk") and name in file.lower():
                        full_path = os.path.join(root, file)
                        try:
                            os.startfile(full_path)
                            print(f"[EXEC] Launched {full_path} from Start Menu")
                            return True
                        except Exception as e:
                            print(f"[EXEC] Failed to launch {full_path}: {e}")

    # 4. Fallback: try spawning via shell if it's safe (simple word)
    try:
        if name.isalnum():
            subprocess.Popen(name, shell=True)
            print(f"[EXEC] Executed shell command: {name}")
            return True
    except Exception:
        pass

    return False


def execute_action(action, input_value, browser=None, deadline=None):
    """Execute a single action. Returns True on success, False on failure.

    open_website/search/youtube_play are considered successful once the
    browser was handed the URL (or the fallback ran); launch_app returns
    the launch_app() result so callers can detect genuine failures.

    *deadline* (F24) is the shared budget handle; the actions that make a
    network request honour it.
    """
    if action == "open_website":
        url = input_value
        if not url.startswith("http"):
            url = "https://" + url

        print(f"[EXEC] Opening: {url} | Browser: {browser}")
        return open_in_browser(url, browser)

    elif action == "launch_app":
        print(f"[EXEC] Launching app: {input_value}")
        if launch_app(input_value):
            return True
        url = input_value
        if not url.endswith((".com", ".net", ".org", ".io")):
            url = url + ".com"
        if not url.startswith("http"):
            url = "https://" + url
        print(f"[EXEC] App launch failed, falling back to website: {url} | Browser: {browser}")
        return open_in_browser(url, browser)

    elif action == "search":
        query = input_value.replace(" ", "+")
        url = f"https://www.google.com/search?q={query}"

        print(f"[EXEC] Searching: {input_value}")
        return open_in_browser(url, browser)

    elif action == "youtube_play":
        print(f"[EXEC] Playing: {input_value}")
        video_url = get_first_youtube_video(input_value, deadline=deadline)

        if video_url:
            return open_in_browser(video_url, browser)
        return open_in_browser(
            f"https://www.youtube.com/results?search_query={input_value}",
            browser,
        )

    return False


def execute_multiple(actions, deadline=None):
    """Execute multiple actions. Returns a list of per-action success bools.

    *deadline* (F24): once the shared budget is spent no further action runs
    (each skipped one reports False, so the result still lines up with the
    input) and the pause between actions is taken from what is left instead
    of granting itself a fresh second.
    """
    print("[EXEC] Executing actions...")

    handle = budget.resolve(deadline)
    results = []
    for action in actions:
        if handle is not None and handle.stopped():
            print("[EXEC] budget exhausted — skipping remaining action(s)")
            results.append(False)
            continue
        print("[EXEC] ->", action)
        try:
            ok = execute_action(
                action.get("action"),
                action.get("input"),
                action.get("browser"),
                deadline=handle,
            )
        except Exception as exc:
            print(f"[EXEC] Action failed: {exc}")
            ok = False
        results.append(bool(ok))
        if handle is not None:
            handle.sleep(BATCH_PAUSE_SECONDS)
        else:
            time.sleep(BATCH_PAUSE_SECONDS)

    print("[EXEC] Execution complete")
    return results
