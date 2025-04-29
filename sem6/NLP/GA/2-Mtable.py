# experiment_2.py
import pandas as pd

# Initialize an empty DataFrame to represent the table
columns = ["Word", "Form", "Tense"]
morph_table = pd.DataFrame(columns=columns)


def add_word_form(df, word, form, tense):
    """Adds a new word form entry to the DataFrame."""
    # Check if the exact entry already exists
    if not df[
        (df["Word"] == word) & (df["Form"] == form) & (df["Tense"] == tense)
    ].empty:
        print(f"Entry already exists: {word}, {form}, {tense}")
        return df

    new_entry = pd.DataFrame([[word, form, tense]], columns=columns)
    df = pd.concat([df, new_entry], ignore_index=True)
    print(f"Added: Word='{word}', Form='{form}', Tense='{tense}'")
    return df


def delete_word_form(df, word=None, form=None, tense=None):
    """Deletes word form entries based on matching criteria."""
    initial_len = len(df)
    conditions = True
    if word:
        conditions &= df["Word"] == word
    if form:
        conditions &= df["Form"] == form
    if tense:
        conditions &= df["Tense"] == tense

    if (
        isinstance(conditions, bool) and conditions
    ):  # Avoid deleting everything if no criteria given
        print("No specific criteria provided for deletion.")
        return df

    rows_to_delete = df[conditions]
    if rows_to_delete.empty:
        print("No matching entries found to delete.")
        return df

    df = df.drop(rows_to_delete.index)
    print(f"Deleted {initial_len - len(df)} entries matching criteria.")
    return df.reset_index(drop=True)  # Reset index after deletion


# --- Example Usage from Description ---

print("Initial Table:")
print(morph_table)
print("-" * 30)

# Add initial data
morph_table = add_word_form(morph_table, "walk", "walks", "Present")
morph_table = add_word_form(morph_table, "talk", "talked", "Past")
morph_table = add_word_form(morph_table, "run", "runs", "Present")

print("\nTable after initial additions:")
print(morph_table)
print("-" * 30)

# Add Operation Example
print("Performing Add Operation:")
morph_table = add_word_form(morph_table, "eat", "eats", "Present")
print("\nTable after adding 'eat':")
print(morph_table)
print("-" * 30)

# Delete Operation Example
print("Performing Delete Operation (removing past tense of 'talk'):")
morph_table = delete_word_form(morph_table, word="talk", form="talked", tense="Past")
# Or more simply if form 'talked' is unique enough:
# morph_table = delete_word_form(morph_table, form="talked")
print("\nTable after deleting 'talked':")
print(morph_table)
print("-" * 30)
