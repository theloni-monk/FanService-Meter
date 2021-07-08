# FanService-Meter: _A Convolutional Neural Network designed to destinguish Hentai from Shonen Manga_
------------------------------
### This project starts with data collection:
* I wrote a web crawler to headlessly find and download manga of the respective catagories
### Training specs:
* The AI was trained in 50 epochs on a Gtx970m gpu
* Reached 99% validation accuracy on a validation set of 257 images
### Utility:
* I wrote a discord bot that can be commanded to analyze a given image on a server and give its output
* The confidence output of the neural net can be used to measure how sexualized a given work of manga is
### Future plans:
* I plan on writing a reddit bot based on the ai that will crawl manga and anime meme subreddits and rate them on how sexualized they are
* If I get around to it I might make a public api for ppl to contribute data and recieve judgements