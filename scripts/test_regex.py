import re

# Comprehensive caption regex supporting:
# 1. Figure 1:, Fig. 1:, Fig 1., Figure 1a:
# 2. Roman numerals: Table I, Table II, TABLE I:
# 3. Unicode spaces: Fig. 1  Caption (no colon or dot)
CAPTION_REGEX = re.compile(
    r"^(?P<type>Figure|Fig\.?|Table)\s*[\xa0\u2002\s]*(?P<num>[0-9IVX]+[a-zA-Z]?)(?:\s*[:\.]|\s+[\u2002\xa0]|\s{2,}|\s+(?=[A-Z]))\s*(?P<caption>.*)",
    re.IGNORECASE | re.DOTALL,
)

test_strings = [
    "Figure 1: Impact and research progress",
    "Fig. 1. Recall@k curves for BM25",
    "TABLE I\nMAIN RETRIEVAL RESULTS",
    "TABLE VI\nFULL RETRIEVAL RESULTS",
    "Fig.\xa01\u2002 Overall architectural diagram of the GraphRAG",
    "Fig.\xa07\u2002 Side-by-side box plots",
    "Table 16: Average generation performance",
    "Table 1: Dataset statistics.",
]

for s in test_strings:
    m = CAPTION_REGEX.match(s.strip())
    if m:
        print(f"MATCH: type={m.group('type')}, num={m.group('num')}, caption={m.group('caption')[:30]}")
    else:
        print(f"FAIL: {repr(s)}")
