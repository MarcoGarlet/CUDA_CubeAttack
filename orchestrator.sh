while true
do
    ./final_attack/offline/maxterm_mining/GPU_version_6/only_gpu/att1 8 20 6 64
	#./final_attack/offline/maxterm_mining/GPU_version_7/only_gpu/att1
    ./final_attack/offline/superpoly_rec/only_gpu/att1
    cat ./final_attack/offline/cubes_test_window.txt >> ./final_attack/offline/cubes_test.txt
    cat ./final_attack/offline/superpolies_window.txt >> ./final_attack/offline/superpolies.txt
	if [[ $(wc -l <./final_attack/offline/cubes_test.txt) -ge 100 ]]; then
		break	
	fi

done

echo "CIPHER EXPLOITED"
