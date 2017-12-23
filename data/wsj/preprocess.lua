-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

return function(word)
   if word ~= '.' and word ~= '~' and word ~= '~~' then
      word = word:gsub('\\', '')
      word = word:gsub('^%[<%S+%]$', '')
      word = word:gsub('^%[%S+>%]$', '')
      word = word:gsub('^%[%S+/%]$', '')
      word = word:gsub('^%[/%S+%]$', '')
      word = word:gsub('^%[%S+%]$', '') -- NOISE
      word = word:gsub('^<(%S+)>$', '%1')
      word = word:gsub('^%*(%S+)%*', '%1')
      word = word:gsub('^%%PERCENT$', 'PERCENT') -- see kaldi
      word = word:gsub('^%.POINT$', 'POINT') -- see kaldi
      word = word:gsub('^%.PERIOD$', '#PERIOD')
      word = word:gsub('^%:COLON$', '#COLON')
      word = word:gsub('^%;SEMI%-COLON$', '#SEMI#COLON')
      word = word:gsub('^%{LEFT%-BRACE$', '#LEFT#BRACE')
      word = word:gsub('^%}RIGHT%-BRACE$', '#RIGHT#BRACE')
      word = word:gsub('^%/SLASH$', '#SLASH')
      word = word:gsub('%.', '#') -- not pronounced but special word in dict
      word = word:gsub('%`', "'") -- typo
      word = word:gsub('^%-HYPHEN$', '#HYPHEN')
      word = word:gsub('^%&AMPERSAND', '#AMPERSAND')
      word = word:gsub('^%?QUESTION%-MARK', '#QUESTION#MARK')
      word = word:gsub('^%(LEFT%-PAREN', '#LEFT#PAREN')
      word = word:gsub('^%)RIGHT%-PAREN', '#RIGHT#PAREN')
      word = word:gsub('^%)CLOSE%-PAREN', '#CLOSE#PAREN')
      word = word:gsub('^%)CLOSE%_PAREN', '#CLOSE#PAREN')
      word = word:gsub('^%(BRACE', '#BRACE')
      word = word:gsub('^%)CLOSE%-BRACE', '#CLOSE#BRACE')
      word = word:gsub('^%(BEGIN%-PARENS', '#BEGIN#PARENS')
      word = word:gsub('^%)END%-PARENS', '#END#PARENS')
      word = word:gsub('^%)END%-THE%-PAREN', '#END#THE#PAREN')
      word = word:gsub('^%)END%-OF%-PAREN', '#END#OF#PAREN')
      word = word:gsub('^%(PAREN', '#PAREN')
      word = word:gsub('^%(PARENTHESES', '#PARENTHESES')
      word = word:gsub('^%)UN%-PARENTHESES', '#UN#PARENTHESES')
      word = word:gsub('^%(IN%-PARENTHESIS', '#IN#PARENTHESES') -- mispell
      word = word:gsub('^%)PAREN', 'PAREN#') -- special case
      word = word:gsub('^,COMMA$', '#COMMA')
      word = word:gsub('^Corp%;$', 'Corp#') -- mispell
      word = word:gsub('^%!EXCLAMATION%-POINT$', '#EXCLAMATION#POINT')
      word = word:gsub('^"QUOTE$', '#QUOTE')
      word = word:gsub('^"DOUBLE%-QUOTE$', '#DOUBLE#QUOTE')
      word = word:gsub('^"IN%-QUOTES$', '#IN#QUOTES')
      word = word:gsub("^'SINGLE%-QUOTE$", '#SINGLE#QUOTE')
      word = word:gsub('^"END%-QUOTE$', '#END#QUOTE')
      word = word:gsub('^"CLOSE%-QUOTE$', '#CLOSE#QUOTE')
      word = word:gsub('^%"END%-OF%-QUOTE', '#END#OF#QUOTE')
      word = word:gsub('^"UNQUOTE$', '#UNQUOTE')
      word = word:gsub('^%-%-DASH$', '##DASH')
      word = word:gsub('^%(PARENTHETICALLY$', 'PARENTHETICALLY') -- special case
      word = word:gsub('%:', '') -- some sort of emphasis
      word = word:gsub('%!', '') -- some sort of whatever
      word = word:gsub('^%-(%S+)$', '%1')
      word = word:gsub('^(%S+)%-$', '%1')
      word = word:gsub('%b()', '') -- not pronounced
      word = word:lower()
      if word:match('%S') then
         local spelling = word:gsub('#', '')
         return word, spelling
      end
   end
end
