from transformers import AutoTokenizer
import load_data
from load_data import initialize_tokenizer


def get_stats(split, tokenizer, processed=False):
    with open(f"data/{split}.nl") as f:
        nl = [line.strip() for line in f]

    with open(f"data/{split}.sql") as f:
        sql = [line.strip() for line in f]

    if processed:
        nl = [preprocess_nl(s) for s in nl]
        sql = [preprocess_sql(s) for s in sql]

    nl_tokens = tokenizer(nl, add_special_tokens=False)["input_ids"]
    sql_tokens = tokenizer(sql, add_special_tokens=False)["input_ids"]

    return [
        ("Number of examples", len(nl), "examples", "d"),
        ("Mean sentence length", sum(map(len, nl_tokens)) / len(nl), "tokens", ".2f"),
        ("Mean SQL query length", sum(map(len, sql_tokens)) / len(sql), "tokens", ".2f"),
        ("Vocabulary size (NL)", len({t for s in nl_tokens for t in s}), "unique tokens", "d"),
        ("Vocabulary size (SQL)", len({t for s in sql_tokens for t in s}), "unique tokens", "d"),
    ]


def preprocess_nl(text):
    return text


def preprocess_sql(sql):
    return (sql
        .replace("<=", " LEQ_OP ")
        .replace(">=", " GEQ_OP ")
        .replace("<", " LT_OP ")
        .replace(">", " GT_OP ")
    )


with open("data_stats.txt", "w") as f:
    for name, finetune in [
        ("Pretrained T5 tokenizer", True),
        ("Custom SentencePiece tokenizer", False),
    ]:
        initialize_tokenizer(finetune)
        tokenizer = load_data.tokenizer

        for processed in [False, True]:
            f.write(
                f"\n{name} - "
                f"{'AFTER' if processed else 'BEFORE'} PREPROCESSING\n"
            )

            for split in ["train", "dev"]:
                f.write(f"\n{split.upper()}\n")
                stats = get_stats(split, tokenizer, processed)

                for key, value, unit, fmt in stats:
                    f.write(f"{key}: {value:{fmt}} {unit}\n")
