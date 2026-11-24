import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from streamlit.testing.v1 import AppTest


def main():
    records = [
        {
            "user_id": f"user-{user}",
            "parent_asin": f"game-{game}",
            "rating": 3 + game % 3,
            "title": "Game review",
            "text": f"Adventure puzzle strategy level number {game}",
        }
        for user, games in enumerate((range(2), range(2, 16), range(1, 16)))
        for game in games
    ]
    with TemporaryDirectory() as tmp:
        fixture = Path(tmp) / "reviews.jsonl"
        fixture.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        os.environ["RECOMMENDER_DATA"] = str(fixture)
        app = AppTest.from_file("app.py", default_timeout=30).run()
        assert not app.exception and not app.error, (app.exception, app.error)
        assert len(app.selectbox) == 1
        app.button[0].click().run()
        assert not app.exception and not app.error, (app.exception, app.error)
        items = [element.value for element in app.markdown if ". Товар " in element.value]
        assert len(items) == 10, items
        assert all("Товар game-0**" not in item and "Товар game-1**" not in item for item in items)


if __name__ == "__main__":
    main()
