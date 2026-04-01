import pyperclip
from playwright.sync_api import sync_playwright, Page
from pathlib import Path

class WhatsAppWebSender:
    def __init__(self, chrome_profile_dir: str):
        self._playwright = sync_playwright().start()
        self._browser    = self._playwright.chromium.launch_persistent_context(
            user_data_dir=chrome_profile_dir,
            channel="chrome",        # usa "msedge" si falla, o quítalo para Chromium puro
            executable_path=self._find_brave_executable(),
            headless=False,
            args=["--profile-directory=Default"],
        )
        self._page = self._browser.pages[0] if self._browser.pages else self._browser.new_page()

    def send_sermon(self, contact: str, audio_path: Path, message: str) -> None:
        page = self._page
        page.goto("https://web.whatsapp.com")
        print("Waiting for WhatsApp Web...")

        page.wait_for_selector('#side', timeout=120_000)
        print("  WhatsApp Web ready.")

        self._open_contact_chat(contact)
        self._send_audio_file(audio_path)
        self._send_text_message(message)
        print("Message sent successfully.")
        page.wait_for_timeout(8000) 

    def close(self) -> None:
        try:
            self._browser.close()
            self._playwright.stop()
        except Exception:
            pass
        
    def _send_audio_file(self, audio_path: Path) -> None:
        page = self._page

        page.locator('button[aria-label="Adjuntar"]').click()
        page.wait_for_timeout(1000)

        # set_input_files works on hidden inputs directly, no click needed
        page.locator('input[type="file"]').first.set_input_files(
            str(audio_path.resolve())
        )
        page.wait_for_timeout(2000)

        send_button = page.locator('div[aria-label="Enviar"][role="button"]')
        send_button.wait_for(state="visible", timeout=15_000)
        send_button.click()

        # Wait for the upload to complete before moving to the text message
        page.wait_for_timeout(5000)


    def _send_text_message(self, message: str) -> None:
        page = self._page

        compose_box = page.locator('div[contenteditable="true"][data-tab="10"]')
        compose_box.wait_for(state="visible", timeout=10_000)

        # Use JavaScript click to bypass any overlay still fading out
        compose_box.evaluate("el => el.click()")

        pyperclip.copy(message)
        page.keyboard.press("Control+V")
        page.wait_for_timeout(500)
        page.keyboard.press("Enter")


    def _open_contact_chat(self, contact: str) -> None:
        page = self._page

        search = page.locator('div[contenteditable="true"][data-tab="3"]')
        search.wait_for(timeout=10_000)
        search.click()
        page.keyboard.type(contact)
        page.wait_for_timeout(2000)

        result = page.locator(f'span[title="{contact}"]').first
        result.wait_for(timeout=10_000)
        result.click()
        page.wait_for_timeout(1500)
        print(f"  Opened chat: {contact}")

    @staticmethod
    def _find_brave_executable() -> str:
        candidates = [
            r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
            r"C:\Program Files (x86)\BraveSoftware\Brave-Browser\Application\brave.exe",
            "/usr/bin/brave-browser",
            "/usr/bin/brave",
        ]
        for path in candidates:
            if Path(path).exists():
                return path
        raise FileNotFoundError("Brave executable not found.")