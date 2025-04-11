"""
Tools to make datasets for training and testing models (for semantic attribute etc.)
"""

import os
from typing import Union, MutableMapping, Mapping, Callable, Optional
from functools import partial

from dol import TextFiles, mk_dirs_if_missing

from mood.util import djoin


# --------------------------------------------------------------------------------------
# Constants

DFLT_SEMANTIC_ATTRIBUTE_DATASET_DIR = djoin("semantic_attributes")

if not os.path.exists(DFLT_SEMANTIC_ATTRIBUTE_DATASET_DIR):
    os.makedirs(DFLT_SEMANTIC_ATTRIBUTE_DATASET_DIR)
    print(f"Created directory: {DFLT_SEMANTIC_ATTRIBUTE_DATASET_DIR}")

# --------------------------------------------------------------------------------------
# Utils

import pathlib
import re
from dol import store_aggregate, cache_iter, TextFiles, DirReader, Files


def text_lines(text: str):
    for line in text.splitlines():
        line = line.strip()
        if line:
            yield line


def mk_concatenation_file(
    dirpath: str,
    *,
    keys_func=sorted,
    savefile_extension=".txt",
    kv_to_item=lambda k, v: "\n".join(text_lines(v)),
    assert_all_chunk_keys=True,
    remove_directory_once_done=False,
):
    # parent of the rootdir
    dirpath = os.path.normpath(dirpath)
    parent_dir = os.path.dirname(dirpath)
    child_dir = os.path.basename(dirpath)
    save_path = os.path.join(parent_dir, child_dir + savefile_extension)
    store = cache_iter(TextFiles(dirpath), keys_cache=keys_func)
    if assert_all_chunk_keys:
        is_chunk_key = lambda k: re.match(child_dir + r"_\d+.*", k)
        assert all(
            map(is_chunk_key, store)
        ), f"Not all keys match the expected pattern: {child_dir}_<number>"
    pathlib.Path(save_path).write_text(
        store_aggregate(
            store,
            kv_to_item=kv_to_item,
            aggregator="\n".join,
        )
    )
    if remove_directory_once_done:
        if len(store) == len(Files(dirpath)):
            import shutil

            shutil.rmtree(dirpath)
        else:
            print(f"Not all files were processed. Keeping {dirpath}")

    return save_path


def parse_line(line: str):
    line_parse_re = re.compile("^(\d+) (.*)$")

    match = line_parse_re.match(line)
    if match:
        score = int(match.group(1))
        segment = match.group(2)
        return dict(score=score, segment=segment)


def parsed_lines(text_contents: str):
    for line in text_lines(text_contents):
        parsed_line = parse_line(line)
        if parsed_line:
            yield parsed_line


# --------------------------------------------------------------------------------------
# Prompt and AI function

dflt_scoring = "A semantic score from 0 to 5 (where 0 means “completely lacks the attribute, or is the opposite of it”, and 5 means “maximally expresses the attribute”). The score should reflect the degree to which the concept is present in the text. You should include examples across the full spectrum — some clearly lacking/opposite, some neutral, ambiguous or mixed, and some highly prototypical."
dflt_facets_spectrum = """
The examples should be **varied across a wide range of linguistic and contextual dimensions** to avoid bias and ensure semantic generality. Specifically, they should differ:

* **In style**  
  (e.g. formality [formal, informal], tone [serious, sarcastic, joyful, melancholic, humorous, threatening], register [colloquial, poetic, academic, bureaucratic, journalistic], sentence complexity [simple, compound, complex], readability [grade-school level to academic prose], and structure [dialogue, monologue, rhetorical question, stream of consciousness])

* **In topic**  
  (e.g. subject matter [politics, science, technology, health, family life, education, war, economics, relationships, religion, art, sports], domain-specific language [legal, technical, medical, business], topicality [current events, timeless topics, historical references], and focus [personal experience, societal issues, abstract ideas])

* **In perspective**  
  (e.g. grammatical person [first-person, second-person, third-person], speaker voice [individual, collective, institutional], ideological stance [liberal, conservative, centrist, apolitical], and point of view [insider, outsider, neutral observer])

* **In emotion or sentiment**  
  (e.g. affective valence [positive, negative, neutral], emotional tone [grief, anger, joy, pride, fear, hope, anxiety, admiration, contempt], intensity [mild, moderate, extreme], and expression style [explicit, understated, metaphorical])

* **In cultural and linguistic context**  
  (e.g. dialect or variety of English [American English, British English, Indian English, African American Vernacular English], idiomatic expression [use of local idioms, metaphors, colloquialisms], cultural references [holidays, traditions, regional events], and worldview framing)

* **In format and genre**  
  (e.g. type of text [tweet, blog post, political speech, news headline, fictional narrative, email, testimonial, FAQ entry], length [short clause, single sentence, multi-sentence paragraph], formatting [list, dialogue exchange, question-answer format])

* **In implied speaker identity or status**  
  (e.g. implied age [child, teenager, adult, elderly], social role [student, parent, official, protester, teacher, employer], power/status tone [subordinate, authoritative, egalitarian], demographic cues [gendered language, class-based references])

* **In temporal framing**  
  (e.g. tense [past, present, future, hypothetical], temporality [historical reference, present moment, envisioned future], and rate of time [momentary, durative, recurring])

* **In communicative intent**  
  (e.g. purpose [informing, persuading, questioning, commanding, storytelling, reflecting, speculating], rhetorical stance [assertive, inquisitive, ironic, pleading, declarative], and audience orientation [self-directed, peer-directed, public address])

This diversity should be naturally embedded in the content and should not be explicitly labeled. The result should be a **rich, heterogeneous dataset** reflecting the wide variability of real-world text expression across all these dimensions.
"""

prompt_template_for_dataset_generation = (
    """" \

You are tasked with generating labeled training data for a machine learning model that 
detects a specific semantic property in text. 
The goal is to produce short text segments {segments_spec:(1–3 sentences each)}
that vary in how much they express the semantic attribute of:
{attribute}.

Please output a list of {n_examples:100} text examples. 
Each example should include:
* short natural language text segment in {language:English}
* {scoring:"""
    + dflt_scoring
    + """}

The score should reflect the degree to which the attribute ({attribute}) is present/reflected/represented
in the text. 

{facets_spectrum:"""
    + dflt_facets_spectrum
    + """}

{score_spectrum:You should include examples across the full spectrum — some clearly not matching the attribute, some ambiguous or mixed, and some highly prototypical.}

The examples should also:
	•   Be varied in topics. {topics: }
	•	Be varied in styles. {styles: }

The output format must be:
	•	One example per line
	•	Each line should have a score, followed by a space, then the text segment (`score text`)
	•	IMPORTANT: No extra commentary, headers, or explanations — only the list

Again, the target attribute is: {attribute}

{extras: }
"""
)

# TODO: Change to more general aix prompt_function when available
import oa

DFLT_TEMPERATURE_FOR_MOOD_DATASET_GEN = 0.7
DFLT_MODEL_FOR_MOOD_DATASET_GEN = "gpt-4o-mini"  # cheapish model

semantic_attribute_examples_for_attribute = oa.prompt_function(
    prompt_template_for_dataset_generation,
    prompt_func=partial(
        oa.chat,
        temperature=float(
            os.environ.get(
                "DFLT_TEMPERATURE_FOR_MOOD_DATASET_GEN",
                DFLT_TEMPERATURE_FOR_MOOD_DATASET_GEN,
            )
        ),
        model=os.environ.get(
            "DFLT_MODEL_FOR_MOOD_DATASET_GEN", DFLT_MODEL_FOR_MOOD_DATASET_GEN
        ),
    ),
)


# def _semantic_attribute_examples_for_attribute(attribute: str, n_examples: int = 100):
#     prompt = prompt_template_for_dataset_generation.format(
#         concept=attribute,
#         n_examples=n_examples,
#     )
#     response = semantic_attribute_examples_for_attribute(prompt)
#     return response


# --------------------------------------------------------------------------------------
# Dataset generation functions


def default_save_key(
    name,
    batch_idx=None,
    *,
    include_subdir_prefix=True,
    counter_key_format="_{:02.0f}",
    extension=".txt",
):
    path_sep = os.path.sep
    suffix = f"{name}{path_sep}" if include_subdir_prefix else ""
    if batch_idx is None:
        return f"{suffix}{name}.txt"
    else:
        counter_str = counter_key_format.format(batch_idx)
        return f"{suffix}{name}{counter_str}{extension}"


def make_semantic_attributes_dataset(
    semantic_attributes: Mapping,
    store: Union[str, MutableMapping] = DFLT_SEMANTIC_ATTRIBUTE_DATASET_DIR,
    *,
    n_examples: int = 1000,
    batch_size: Optional[int] = 100,
    start_batch_idx_at: int = 0,
    save_key: Callable = default_save_key,
    verbose: int = 0,
    **dataset_maker_kwargs,
):
    """
    Gather data for each semantic attribute and save it to the store.

    Parameters:
    - semantic_attributes: Mapping of attribute names to their descriptions
    - store: Storage location (string path or MutableMapping)
    - n_examples: Total number of examples to gather per attribute
    - batch_size: Number of examples per batch; if None, gather all examples in one batch
    - start_batch_idx_at: Starting index for batch numbering
    - save_key: Function to generate save keys
    - verbose: Verbosity level (0=quiet, 1=attribute progress, 2=batch progress)
    - dataset_maker_kwargs: Additional kwargs for semantic_attribute_examples_for_attribute

    WARNING: Some models aren't very good at generating the requested number of examples.
    They may generate fewer or more than requested. This is a known issue with LLM
    models.

    >>> store = {}
    >>> attributes = {"color": "description of color"}
    >>> make_semantic_attributes_dataset(attributes, store, n_examples=3, batch_size=2)
    >>> len(store)  # Two batches saved
    2
    >>> store  # doctest: +SKIP
    {
        'color/color_00.txt': '0 The meeting was dull and unproductive, leaving everyone feeling dissatisfied.'
                              '2 The car was old and rusty, with a paint job that had faded over time, hinting at its once vibrant color.'
                              '5 The sunset painted the sky in brilliant shades of orange, pink, and purple, creating a breathtaking display of color that captivated all who watched.'
                              '1 The room felt gloomy, with dark furniture casting shadows, though a hint of blue could be seen in the curtains.  ',
        'color/color_01.txt': '0 The meeting was dull and uneventful, lacking any excitement or engagement.'
                              '1 The car was just an ordinary vehicle, nothing special about its appearance. '
                              "2 The flowers were vibrant, but I couldn't quite tell what colors they were. "
                              '3 The sky turned a beautiful shade of orange as the sun began to set, casting a warm glow.'
                              "4 She wore a stunning red dress that caught everyone's attention at the party."
                              '5 The ocean sparkled in shades of deep blue and turquoise, a breathtaking view that left us speechless.  '
    }

    """
    if isinstance(store, str):
        store = mk_dirs_if_missing(TextFiles(store))

    def _get_batch_config(n_examples, batch_size, start_idx):
        """
        Generate batch configuration based on whether batch_size is None or not.
        Returns a generator of (batch_idx, current_batch_size) tuples.
        """
        if batch_size is None:
            # Single batch mode
            yield None, n_examples
        else:
            # Multiple batches mode
            num_batches = (
                n_examples + batch_size - 1
            ) // batch_size  # Ceiling division
            for i in range(num_batches):
                batch_idx = start_idx + i
                remaining = n_examples - i * batch_size
                current_batch_size = min(batch_size, remaining)
                yield batch_idx, current_batch_size

    for attribute_idx, (name, attribute) in enumerate(semantic_attributes.items(), 1):
        if verbose > 0:
            print(f"{attribute_idx}/{len(semantic_attributes)}: {name}")

        for batch_idx, current_batch_size in _get_batch_config(
            n_examples, batch_size, start_batch_idx_at
        ):
            _save_key = save_key(name, batch_idx)

            if verbose > 1:
                print(f"  Gathering {current_batch_size} examples for: {_save_key}")

            response = semantic_attribute_examples_for_attribute(
                attribute=attribute,
                n_examples=current_batch_size,
                **dataset_maker_kwargs,
            )
            store[_save_key] = response
