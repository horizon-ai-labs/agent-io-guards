#!/bin/bash
# exit 0 when none of the given job ids is reserved/running; exit 1 otherwise or if status is unavailable
S=$(compute status 2>/dev/null)
echo "$S" | grep -q '"jobs"' || exit 1
for j in "$@"; do echo "$S" | grep -q "\"$j[^\"]*\", \"state\": \"\(reserved\|running\)\"" && exit 1; done
exit 0
