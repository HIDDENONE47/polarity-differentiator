"""
diagnose_tavily.py

One-off diagnostic for the agent_auditor.py verification bug.

Run this from inside your data_pipeline/ folder (same place you run
graph_orchestrator.py from), using the same virtual environment / installed
packages as the rest of the pipeline. It needs nothing new -- tavily-python
and python-dotenv are already pipeline dependencies.

What it checks:
1. That TAVILY_API_KEY is actually being loaded from .env.
2. That client.search() still works (a known-good baseline -- this is the
   call discover_node already uses successfully, so it tells us if the
   problem is account-wide or extract-specific).
3. That client.extract() works on a real, easy URL (Wikipedia, as a control)
   AND on one of your own dataset's actual source URLs.
4. Whether extract() throws an exception, OR silently returns an empty
   'results' list while reporting the real failure in 'failed_results'
   instead -- a second failure mode agent_auditor.py currently never
   checks for at all (it only ever looks at result.get("results")).
"""

import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

api_key = os.getenv("TAVILY_API_KEY")
if api_key:
    print(f"TAVILY_API_KEY loaded, ends in ...{api_key[-4:]}")
else:
    print("TAVILY_API_KEY NOT FOUND -- check your .env file and that you're running this from data_pipeline/")

client = TavilyClient(api_key=api_key)

print("\n=== 1. client.search() baseline (known-working call) ===")
try:
    r = client.search(query="family office investment", max_results=1)
    print("OK -- search() returned", len(r.get("results", [])), "result(s)")
except Exception as e:
    print("FAILED:", type(e).__name__, "-", e)

test_urls = [
    "https://en.wikipedia.org/wiki/Family_office",  # control: should be trivially extractable if anything is
    "https://www.partnerspath.com",                  # a real source_url from your own dataset
]

print("\n=== 2. client.extract() on test URLs ===")
for url in test_urls:
    print(f"\n-> {url}")
    try:
        result = client.extract(urls=[url])
        pages = result.get("results", [])
        failed = result.get("failed_results", [])
        if pages:
            content = pages[0].get("raw_content", "")
            print(f"   SUCCESS -- got {len(content)} chars of raw_content")
        else:
            print("   NO EXCEPTION, but 'results' was empty.")
            print(f"   'failed_results': {failed}")
    except Exception as e:
        print(f"   EXCEPTION: {type(e).__name__} - {e}")

print("\nDone. Paste everything above back to Claude.")
