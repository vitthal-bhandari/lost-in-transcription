# id_jv data samples (random, seed=7)

Format: `[filename | duration] transcript`. Train transcripts are raw (pre-normalization,
pre-diacritic-fold) — note the pepet/taling marks (ê, è) absent from dev.

## Train (Jember corpus, 10 random)

```
[202.mp3 | 8.0s] Ya, sakjane uduk masalah kurange lembaga pendhidhikan ya, uduk masalah sêkolahane.
[31.mp3 | 28.0s] Tujuane bên isa ngimbangi antara wong sing, e, fanatik ndèk agama tok karo fanatik ndèk pendhidhikan formal.
[62.mp3 | 7.0s] Kadang lèk aku isieg-iseng kan aku ngêrti basa Madura sithik-sithik, dadine ya ngobrol karo basa Madura.
[196.mp3 | 4.0s] Dadi sikile ya miring ngono lèk mandhêg.
[97.mp3 | 19.0s] S..., saat ana nang panggonan wisata aku luwih sêring tuku makanan.
[117.mp3 | 8.0s] Tapi umpama pas KKN, budhal nèng Prau iku kan, e, variasine ora pati, ora pati akèh kaya nang Jogja, tapi menarik.
[10.mp3 | 9.0s] Soale beberapa taun iki suwi-suwi ki tambah panas, dibandingna, e, panase iku nêmên, dibandingna taun lalu.
[29.mp3 | 3.0s] De'e iku, duwek iku ya wis ana ae.
[106.mp3 | 8.0s] Karêpe iku, e..., nggak njupuk sing berat-berat ngono i, apa, sing ana parameter berat sing kudu dikêthok iku.
[24.mp3 | 5.0s] Soale aku saiki apa ya, ndak menargetkan apa-apa kaya misal.
```

## Dev (official indonesian_dev, 10 random)

```
[db2fbac8b35449ff81fb51d31fb167cc.mp3] Hm.. ya..., kalau dipikir seru banget ya, maksudnya kayak bekerja sama tapi mengerjakan PR, ya meskipun sebagian agak nyontek-nyontek 'kan, tapi kan dulu kita ngerjain juga nggak pasti full plek mencontek. Paling juga, apa namanya, kalau misal yang isian gitu yang pasti jawabannya baru nyontek. ya nggak sih? Tapi seru kayak tukeran, jadi kayak barter.
[1635a2c074f2492db7964daf8e83694b.mp3] Dan hal kuwi membuatku senang, artine apa? Artine kan hobi kan iki, ya cuma sarana kita untuk melepas penat dan stres ya. Jadi ya, ya hobi ya... butuh apa wae. Hobi yang, menurutku hobi yang bermanfaat i gak... gak iso deh. Ana sing kan ngomong ya, beberapa influencer, carilah hobi yang menghasilkan, carilah hobi yang bermanfaat, carilah hobi yang membu.. anu yang... yang menjadi cuan.
[2289f904703947a5babb0fd172580555.mp3] Terus nek dari segi pembelajaran pas SMA tuh jadi makin sering ya tugas-tugas atau PR gitu pakai ... kudu pakai teknologi internet atau ngetik di laptop atau komputer gitu ya buat belajar. Nah, biasane sih, paling sering, ya tugas presentasi kelompok. Nganggo powerpoint opo tugas bikin esai. Nah, pas SMA ini aku jadi perlu atau sering banget pakai laptop untuk ngerjain tugas.
[03f2aa9e6a44423fbad778cddbd52645.mp3] Pokokmen sing penting buat apa ya self reward gitu apa jalan-jalan gitu, harus yang emang aku seneng dan gak memberatkan juga tentunya gak bikin pekerjaanku jadi keteteran. Jadi pokoknya aku harus bisa ngatur waktu juga supaya gak terlalu fokus seneng-senengnya gitu tapi tetap harus fokus kerja juga.
[1c49695fbef347e0b2b6cd53c9f01136.mp3] Biasane pelajaran yang paling angel ki ekonomi karo matematika. Nah, sing pinter ekonomi ngajari liyane, sing pintar matematika ngajari sing liyane. Aku termasuk sing diajari, sih, udu sing ngajari, haha.
[cb245ad315ec40caaf6a83a662aa756b.mp3] Karena aku suka film sama drama Jepang, ya biasanya aku nonton e.. beberapa judul itu dari Jepang. Terus yak.. nontonnya paling biasanya di Netflix opo nggak download dari internet gitulah. Jadi ini semacam.. jadi semacam reward gitu, hadiah, nek aku habis ngerjain.. nyelesein.. setelah nyelesein kerjaan yang banyak.
[e645170bb55f4b80a2828e19a831e918.mp3] Oke, cerita belajar pas SMP. Jadi belajar di SMP itu menurutku adalah masa di mana aku mulai belajar pakai teknologi. Ya, kayak ngetik-ngetik pakai komputer, apa cari materi di internet, terus presentasi pakai powerpoint juga.
[8f93952bc7534c7aa4e70ec90299b739.mp3] Selain makanan baru, aku ning kene ki juga dapat hobi baru. Nah hobiku baru adalah lari atau jogging. Sekilas ki hobi jogging kayake ya... gur gampang ngono ya, murah juga ngono lho. Ya gur butuh sikil, butuh niat, mlaku, mlayu, gas, wes rampung. Ternyata, ketika aku mendalami hobi itu semakin dalam, semakin lama, ternyata gak niat thok sek dibutuhke.
[dfaef032d0924570a785f64d1f4b13ba.mp3] Ini kayaknya aku kalau ke tempatmu akan mulai mempertimbangkan lewat dari arah lain deh. Gak tahu deng, berdzikir sajalah banyak-banyak.
[7917eb216a8b43c3ae9a2ebc3e08e51f.mp3] Terus ya akhirnya alhamdulillah aku bisa dapat nilai yang lumayan pas UN, pas ujian nasional. Dan, alhamdulillah bisa masuk SMA negeri yang ya terbilang favorit gitu ya di kotaku. Terus ... di SMA pengalamanku belajar ya udah beda lagi daripada SD sama SMP. Nanti tak sambung meneh ceritane. Kalau Mas Bondan gimana pengalaman pas SMP dulu?
```

## Observations for the next agent
- Train segments are short (median ~5–8s here, up to 28s); dev utterances run noticeably longer
  and more continuous (no hard 40s cap observed in this sample, several 60+ words).
- Train transcripts carry dense pepet/taling diacritics (ê, è) — none appear in this dev sample.
- Train skews more Javanese-forward in register; dev reads more Indonesian-forward with Javanese
  code-switching woven in (`iki`, `sing`, `ngerjain`, `piye` alongside standard Indonesian).
