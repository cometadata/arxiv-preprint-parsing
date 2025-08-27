from pydantic import BaseModel, RootModel, Field
from typing import List, Annotated
import json

class Author(BaseModel):
    """
    A single author entry with one or more institutional affiliations.
    """

    name: str = Field(
        ...,
        description=(
            "Author's full name, exactly as it should appear in the publication "
            "(e.g., 'Naser Damer')."
        ),
    )
    affiliations: List[str] = Field(
        ...,
        description=(
            "Ordered list of the author’s institutional affiliations. "
            "Each item should be a human-readable string such as "
            "'Fraunhofer Institute for Computer Graphics Research IGD, Darmstadt, Germany'."
        ),
    )


class Affiliations(
    RootModel[
        Annotated[
            List[Author],
            Field(
                description="List of authors in the exact order they appear on the paper."
            ),
        ]
    ]
):
    """
    Top-level model that wraps the array of authors.
    Using RootModel lets Pydantic treat a bare JSON list as a single model.
    """

EXAMPLE_OUTPUT = [
    {
        "affiliations": [
            "School of Mathematical and Computational Sciences North Haugh, St Andrews, Fife KY16 9SS, UK"
        ],
        "name": "M.D. Atkinson"
    },
    {
        "affiliations": [
            "Department of Mathematics University College, Galway, Eire"
        ],
        "name": "G. Pfeiffer"
    }
]

SYSTEM_PROMPT = f"""You are an expert at reading academic articles and parsing information about their affiliations. The user will show you an academic article and your job is to extract the authors and their affiliations in a structured format.

### JSON Schema

{json.dumps(Affiliations.model_json_schema(), indent=2)}

### Example Output

{json.dumps(EXAMPLE_OUTPUT, indent=2)}

### Summary

Read the article carefully, paying attention to the authors and their affiliations. Then respond with a JSON object in the format specified above that contains the authors and their affiliations."""
