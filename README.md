# SongSuro

**SongSuro** is an AI project to swap the lyrics from the original song.
It keeps the song style, pitch, rhythm, and melody unchanged, but only the lyrics are swapped to your input lyrics.

## Demo

<Demo will be here>

[Our YouTube Channel](https://www.youtube.com/@RiceBobbFoundation)


## Build from source

### 1. Install uv
```bash
# UNIX, Mac, Linux, WSL
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Configure uv environment

In Pycharm, you can set a local interpreter using uv (need to update Pycharm to the latest version)


### 3. Install pre-commit
Install pre-commit to use ruff as linter and reformat.

```
pre-commit install
```

## Test
We use pytest for testing.

```bash
python3 -m pytest -n auto tests/
```

## 🎯 Troubleshooting
### 0️⃣ LookupError: `Resource cmudict not found` / `Resource averaged_perceptron_tagger not found`
```markdown
E LookupError:
E **********************************************************************
E Resource cmudict not found.
E Please use the NLTK Downloader to obtain the resource:
E
E >>> import nltk
E >>> nltk.download('cmudict')
E
E For more information see: https://www.nltk.org/data.html
E
E Attempted to load corpora/cmudict
```

This error occurs when the `cmudict` resource is not installed in your NLTK data directory.
Try running the following command in your Python environment to download the `cmudict` resource:

```python
import nltk
nltk.download('cmudict')
```

### cf) Resource `blahblah` not found
You may see a similar error message for other resources, such as `blahblah` or `averaged_perceptron_tagger`.

```markdown
E LookupError:
E **********************************************************************
E Resource blahblah not found.
E Please use the NLTK Downloader to obtain the resource:
E
E >>> import nltk
E >>> nltk.download('blahblah')
E
E For more information see: https://www.nltk.org/data.html
E
E Attempted to load corpora/blahblah
```

You can use the same command to download the resource that is not found.

```python
import nltk
nltk.download('blahblah')
```


### 1️⃣ Error: `SSL error downloading NLTK data`
The error message is like this:
```markdown
[nltk_data] Error loading cmudict: <urlopen error
[SSL: [nltk_data] CERTIFICATE_VERIFY_FAILED] certificate verify failed:
[nltk_data] unable to get local issuer certificate (_ssl.c:1000)>
```

or

```markdown
[nltk_data] Error loading averaged_perceptron_tagger: <urlopen error
[nltk_data]     [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify
[nltk_data]     failed: unable to get local issuer certificate
[nltk_data]     (_ssl.c:1007)>
import nltk
```

### [Way 1](https://github.com/myshell-ai/MeloTTS/issues/153) Disable SSL certificate verification
Try this first.

```Python
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
```

This code snippet will disable SSL certificate verification for the NLTK downloader, allowing it to download resources without encountering SSL errors.

And run the following command to download the your_nltk resource again.:

```python
import nltk
nltk.download('YOUR_RESOURCE')
```

### [Way 2](https://stackoverflow.com/questions/41348621/ssl-error-downloading-nltk-data/42890688#42890688) Install certificates

Run the following command to install the certificates. Please replace `Python 3.x` with your Python version.

```bash
/Applications/Python 3.x/Install Certificates.command
```
Refers this
- [this](https://github.com/nltk/nltk/issues/2029)
- [this](https://stackoverflow.com/questions/41348621/ssl-error-downloading-nltk-data/42890688#42890688)
