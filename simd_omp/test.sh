(./a.out) > output2.txt
(../sequential/a.out) > output1.txt
restult=$(diff output1.txt output2.txt)
if [ $? -ne 0 ]
then
  echo "Conv FAILED"
else
  echo "Conv PASSED"
fi
# restult=$(diff out_fc.txt ../sequential/out_fc.txt)
# if [ $? -ne 0 ]
# then
#   echo "Full FAILED"
# else
#   echo "Full PASSED"
# fi
# restult=$(diff out_soft.txt ../sequential/out_soft.txt)
# if [ $? -ne 0 ]
# then
#   echo "Soft FAILED"
# else
#   echo "Soft PASSED"
# fi
# restult=$(diff out_pool.txt ../sequential/out_pool.txt)
# if [ $? -ne 0 ]
# then
#   echo "Pool FAILED"
# else
#   echo "Pool PASSED"
# fi
