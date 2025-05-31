import glob
import json
import os
import asyncio

import click
from dotenv import load_dotenv
from openai import AsyncOpenAI
import pandas as pd

from songsuro.utils.util import process_batch


def build_lyrics_csv(root_dir: str, save_path: str, batch_size: int = 32):
	client = AsyncOpenAI()

	json_filepaths = []
	for json_file in glob.iglob(
		os.path.join(root_dir, "라벨링데이터", "**", "*.json"), recursive=True
	):
		json_filepaths.append(json_file)

	output_dict_list = []
	for json_filepath in json_filepaths:
		label_data = json.load(open(json_filepath, "r"))
		phoneme_list = list(
			map(lambda x: x["lyric"] if x["lyric"] else " ", label_data["notes"])
		)
		phoneme = "".join(phoneme_list)
		output_dict_list.append(
			{
				"json_filepath": json_filepath,
				"filename": os.path.basename(json_filepath),
				"phoneme": phoneme,
			}
		)

	loop = asyncio.get_event_loop()
	tasks = [get_grapheme(client, x["phoneme"]) for x in output_dict_list]
	result = loop.run_until_complete(process_batch(tasks, batch_size=batch_size))

	df = pd.DataFrame(output_dict_list)
	df["grapheme"] = result

	df.to_csv(save_path, index=False)


async def get_grapheme(client, phoneme: str):
	system_prompt = """
You are an AI assistant to change Korean phoneme lyrics to the grapheme.
The phoneme is the pronunciation of the lyrics, yet grapheme is the lyrics that human can easily understand.
You must not return the grapheme lyrics only, not the other word.
The user will give you only phoneme lyrics, not other explaination.
The result grapheme lyrics is in Korean, obviously.
Here is the user’s input (phoneme lyrics)."""

	completion = await client.chat.completions.create(
		model="gpt-4.1",
		messages=[
			{"role": "developer", "content": system_prompt},
			{"role": "user", "content": phoneme},
		],
	)
	output_grapheme = completion.choices[0].message.content
	return output_grapheme


@click.command()
@click.option(
	"--root_dir", type=click.Path(exists=True, dir_okay=True, file_okay=False)
)
@click.option("--save_path", type=click.Path(exists=False))
@click.option("--batch_size", type=int, default=32)
def cli(root_dir: str, save_path: str, batch_size: int = 32):
	build_lyrics_csv(root_dir, save_path, batch_size)


if __name__ == "__main__":
	load_dotenv()
	cli()
