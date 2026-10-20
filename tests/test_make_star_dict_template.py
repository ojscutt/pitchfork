import json
import unittest
from pathlib import Path


class MakeStarDictNotebookTest(unittest.TestCase):
    def test_template_dict_is_assigned_to_star_dict(self):
        notebook_path = Path(__file__).resolve().parents[1] / "0-make-star-dict.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        template_cell = next(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
            and "star_name = ..." in "".join(cell.get("source", []))
        )
        self.assertRegex(
            template_cell,
            r"(?m)^star_dict\s*=",
            "template must assign the dictionary to star_dict",
        )


if __name__ == "__main__":
    unittest.main()
