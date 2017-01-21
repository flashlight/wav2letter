for line in io.lines("toto") do
   os.execute("rm -f " .. line)
   print(line)
end
