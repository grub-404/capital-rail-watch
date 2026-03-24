Editing "Did you know?" lines for the ticker
============================================

1. Open this file in the overlay folder:
      ticker-facts.json

2. You will see a list under "facts": each line is in "quotes" and ends with a comma,
   except the last one (no comma after the last fact).

3. To add a fact: copy a line including the quotes and comma, paste it before the
   closing bracket ], and edit the text inside the quotes.

4. Save the file. The ticker reloads this file whenever it refreshes train data
   (about every minute by default), or refresh the browser / OBS browser source
   to see changes right away.

5. If the file has a mistake (invalid JSON), the ticker will keep using the
   built-in default facts until the file is fixed.

Tip: Use a normal text editor. If your editor offers "JSON" validation, turn it on
to catch missing commas or quotes.
