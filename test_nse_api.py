import datetime as dt
from build_adjusted_master import fetch_nse_corporate_actions

fd = dt.date.today() - dt.timedelta(days=45)
td = dt.date.today() + dt.timedelta(days=15)

from_s = fd.strftime("%d-%m-%Y")
to_s = td.strftime("%d-%m-%Y")

print(f"Fetching from {from_s} to {to_s}...")
res = fetch_nse_corporate_actions(from_s, to_s)
print(f"Got {len(res)} results.")
if res:
    print("Sample:", res[0])

