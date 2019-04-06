export fileid=1fneFKxF7bPqc1fFCvQ5M678VcWgUyz8e
export filename=andhrajyothy.zip

wget --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id='$fileid -O- \
	| sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt

wget --load-cookies cookies.txt -O $filename \
	'https://docs.google.com/uc?export=download&id='$fileid'&confirm='$(<confirm.txt)

rm -f confirm.txt cookies.txt

# export fileid=1eAdP8K6IPXZ26yy-ayTyRWRbcvAr6QCG
# export filename=webdunia.zip
# export fileid=1ICvkRxz1j6_YvDTOpN2ytGJ4snvNq0T8
# export filename=wiki.zip