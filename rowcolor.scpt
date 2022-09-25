#!/usr/bin/env osascript

(** select certain rows in Apple Numbers **)

set outEps to {65024, 65024, 52992}
set outInst to {64256, 56576, 55296}
set outAll to {48640, 55808, 55296}

tell application "Numbers"
     activate
     tell document 1 to tell sheet "stats" to tell table 1
     	  repeat with i from 1 to row count
	  	 set l to value of cell 1 of row i
	  	 if l = "OUTEPS:-1"
		    -- set selection range to row i
		    set background color of row i to outEps
		 else if l = "OUTINST:-1"
		    set background color of row i to outInst
		 else if l = "OUTALL:-1"
		    set background color of row i to outAll
	  	 end if
          end repeat
     end tell
end tell

