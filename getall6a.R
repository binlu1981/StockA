library("quantmod")
source("GetFinancialInfo.netease.R")

type.list <- c("zycwzb", "cznl", "ylnl")
stocksA<-read.table("A3a.txt",header=T)
for (i in stocksA[,"Codes"]) {
	ID=substr(i,1,6)
#	STOCKNAME=toString(stocksA[rownames(subset(stocksA,Codes==i)),2])
	setSymbolLookup(STOCKNAME=list(name=i,src="yahoo"))
	tryit<-try(getSymbols("STOCKNAME"))
	if(inherits(tryit, "try-error")) {
		i<-stocksA[,"Codes"][match(i,stocksA[,"Codes"])+1]				 
	} else {
		getSymbols("STOCKNAME")
	}
#   toString
	to.yearly(STOCKNAME)
	yearlyReturn(STOCKNAME)
	STOCKNAMErmna<-na.omit(STOCKNAME)
	lastr=nrow(to.yearly(STOCKNAMErmna))	
	# if (lastr>=5) {
		# STOCKNAME3year<-as.matrix(c(STOCKNAMErmna[c('2013-07-26'),],tail(STOCKNAMErmna,1)))
		# StockPriceIncreaseRaito3year<-(STOCKNAME3year[2,6]-STOCKNAME3year[1,6])/STOCKNAME3year[1,6]
	# } else {
		# STOCKNAME3year<-as.matrix(c(STOCKNAME[1,],tail(STOCKNAME,1)))
		# StockPriceIncreaseRaito3year<-(STOCKNAME3year[2,6]-STOCKNAME3year[1,6])/STOCKNAME3year[1,6]
	# }
	if (nrow(STOCKNAMErmna)>750) {
		STOCKNAME3year<-as.matrix(c(STOCKNAMErmna[nrow(STOCKNAMErmna)-750,],tail(STOCKNAMErmna,1)))
	} else {
		STOCKNAME3year<-as.matrix(c(STOCKNAME[1,],tail(STOCKNAME,1)))
	}
	StockPriceIncreaseRaito3year<-(STOCKNAME3year[2,6]-STOCKNAME3year[1,6])/STOCKNAME3year[1,6]		
	if (nrow(STOCKNAMErmna)>500) {
		STOCKNAME2year<-as.matrix(c(STOCKNAMErmna[nrow(STOCKNAMErmna)-500,],tail(STOCKNAMErmna,1)))
	} else {
		STOCKNAME2year<-as.matrix(c(STOCKNAME[1,],tail(STOCKNAME,1)))
	}
	StockPriceIncreaseRaito2year<-(STOCKNAME2year[2,6]-STOCKNAME2year[1,6])/STOCKNAME2year[1,6]		
	if (nrow(STOCKNAMErmna)>250) {
		STOCKNAME1year<-as.matrix(c(STOCKNAMErmna[nrow(STOCKNAMErmna)-250,],tail(STOCKNAMErmna,1)))
	} else {
		STOCKNAME1year<-as.matrix(c(STOCKNAME[1,],tail(STOCKNAME,1)))
	}
	StockPriceIncreaseRaito1year<-(STOCKNAME1year[2,6]-STOCKNAME1year[1,6])/STOCKNAME1year[1,6]
	if (nrow(STOCKNAMErmna)>125) {
		STOCKNAME6month<-as.matrix(c(STOCKNAMErmna[nrow(STOCKNAMErmna)-125,],tail(STOCKNAMErmna,1)))
	} else {
		STOCKNAME6month<-as.matrix(c(STOCKNAME[1,],tail(STOCKNAME,1)))
	}
	StockPriceIncreaseRaito6month<-(STOCKNAME6month[2,6]-STOCKNAME6month[1,6])/STOCKNAME6month[1,6]
	if (nrow(STOCKNAMErmna)>60) {
		STOCKNAME3month<-as.matrix(c(STOCKNAMErmna[nrow(STOCKNAMErmna)-60,],tail(STOCKNAMErmna,1)))
	} else {
		STOCKNAME3month<-as.matrix(c(STOCKNAME[1,],tail(STOCKNAME,1)))
	}
	StockPriceIncreaseRaito3month<-(STOCKNAME3month[2,6]-STOCKNAME3month[1,6])/STOCKNAME3month[1,6]
	if (nrow(STOCKNAMErmna)>20) {
		STOCKNAME1month<-as.matrix(c(STOCKNAMErmna[nrow(STOCKNAMErmna)-20,],tail(STOCKNAMErmna,1)))
	} else {
		STOCKNAME1month<-as.matrix(c(STOCKNAME[1,],tail(STOCKNAME,1)))
	}
	StockPriceIncreaseRaito1month<-(STOCKNAME1month[2,6]-STOCKNAME1month[1,6])/STOCKNAME1month[1,6]

	Open<-toString(STOCKNAMErmna[nrow(STOCKNAMErmna),1])
	High<-toString(STOCKNAMErmna[nrow(STOCKNAMErmna),2])
	Low<-toString(STOCKNAMErmna[nrow(STOCKNAMErmna),3])
	Close<-toString(STOCKNAMErmna[nrow(STOCKNAMErmna),4])
	Volume<-toString(STOCKNAMErmna[nrow(STOCKNAMErmna),5])
	Adjusted<-toString(STOCKNAMErmna[nrow(STOCKNAMErmna),6])
	for (ti in type.list) {
		
		GetFinancialInfo.netease(ID,translate=FALSE, clean=T, quiet=TRUE, savefile=NA, type=ti)
			}	
			if (substr(i,1,1)=="6") {
				profitincrease<-get(paste("SH",ID,"CZNL",sep=""))
				fincondition<-get(paste("SH",ID,"ZYCWZB",sep=""))
				interestrate<-get(paste("SH",ID,"YLNL",sep=""))
			} else {
				profitincrease<-get(paste("SZ",ID,"CZNL",sep=""))
#				as.name
				fincondition<-get(paste("SZ",ID,"ZYCWZB",sep=""))
				interestrate<-get(paste("SZ",ID,"YLNL",sep=""))
			}	
			profitincreasermna<-na.omit(profitincrease[,c(1,3)])			
			if (nrow(profitincreasermna)>=13) {
				profitincrease3yearrmna<-as.matrix(profitincreasermna[1:13,])
				
				if (as.numeric(profitincrease3yearrmna[1,2])>=as.numeric(profitincrease3yearrmna[5,2])) {
					if (as.numeric(profitincrease3yearrmna[5,2])>=as.numeric(profitincrease3yearrmna[9,2])) {
						if (as.numeric(profitincrease3yearrmna[9,2])>=as.numeric(profitincrease3yearrmna[13,2])) {
							profitgrowth_increase<-"profitgrowth_up_3year"
						}
					}
				}	else {
							profitgrowth_increase<-"profitgrowth_down_3year"
				}
			} else {
				profitincrease3yearrmna<-as.matrix(profitincreasermna)
				if (as.numeric(profitincrease3yearrmna[1,2])>=as.numeric(profitincrease3yearrmna[nrow(profitincrease3yearrmna),2])) {
							profitgrowth_increase<-"profitgrowth_up_1-2year"
				}	else {
							profitgrowth_increase<-"profitgrowth_down_1-2year"
				}
			}			
			finconditionrmna<-na.omit(fincondition[,c(1,2,11)])
			if (nrow(finconditionrmna)>=13)	{
				fincondition3year<-as.matrix(fincondition[1:13,c(1:3,11:18,20)])			
				fincondition3yearrmna<-na.omit(fincondition3year[,c(1,8,10)])
				fincondition3year2p<-as.matrix(finconditionrmna[c(1,13),])
				profitincreaseratio3y<-(as.numeric(fincondition3year2p[1,3])-as.numeric(fincondition3year2p[2,3]))/as.numeric(fincondition3year2p[2,3])
			}	else {
				fincondition3year<-as.matrix(fincondition[,c(1:3,11:18,20)])
				fincondition3yearrmna<-na.omit(fincondition3year[,c(1,8,10)])
				fincondition3year2p<-as.matrix(finconditionrmna[c(1,nrow(finconditionrmna)),])
				profitincreaseratio3y<-(as.numeric(fincondition3year2p[1,3])-as.numeric(fincondition3year2p[2,3]))/as.numeric(fincondition3year2p[2,3])
			}
			if (nrow(finconditionrmna)>=9)	{
				fincondition2year<-as.matrix(fincondition[1:9,c(1:3,11:18,20)])			
				fincondition2yearrmna<-na.omit(fincondition2year[,c(1,8,10)])
				fincondition2year2p<-as.matrix(finconditionrmna[c(1,9),])
				profitincreaseratio2y<-(as.numeric(fincondition2year2p[1,3])-as.numeric(fincondition2year2p[2,3]))/as.numeric(fincondition2year2p[2,3])
			}	else {
				fincondition2year<-as.matrix(fincondition[,c(1:3,11:18,20)])
				fincondition2yearrmna<-na.omit(fincondition2year[,c(1,8,10)])
				fincondition2year2p<-as.matrix(finconditionrmna[c(1,nrow(finconditionrmna)),])
				profitincreaseratio2y<-(as.numeric(fincondition2year2p[1,3])-as.numeric(fincondition2year2p[2,3]))/as.numeric(fincondition2year2p[2,3])
			}
			if (nrow(finconditionrmna)>=5)	{
				fincondition1year<-as.matrix(fincondition[1:5,c(1:3,11:18,20)])			
				fincondition1yearrmna<-na.omit(fincondition1year[,c(1,8,10)])
				fincondition1year2p<-as.matrix(finconditionrmna[c(1,5),])
				profitincreaseratio1y<-(as.numeric(fincondition1year2p[1,3])-as.numeric(fincondition1year2p[2,3]))/as.numeric(fincondition1year2p[2,3])
			}	else {
				fincondition1year<-as.matrix(fincondition[,c(1:3,11:18,20)])
				fincondition1yearrmna<-na.omit(fincondition1year[,c(1,8,10)])
				fincondition1year2p<-as.matrix(finconditionrmna[c(1,nrow(finconditionrmna)),])
				profitincreaseratio1y<-(as.numeric(fincondition1year2p[1,3])-as.numeric(fincondition1year2p[2,3]))/as.numeric(fincondition1year2p[2,3])
			}
			EPS<-toString(finconditionrmna[grepl("-12-", finconditionrmna[,1]),][1,2])
			PB<-toString(as.numeric(Close)/as.numeric(fincondition3year[1,3]))
			PE<-as.numeric(Close)/as.numeric(EPS)
			ROE<-toString(fincondition3year[1,12])
			Net_profit<-toString(fincondition3year[1,4])
			cash_flow<-toString(fincondition3year[1,6])
			Current_assets<-toString(fincondition3year[1,9])
			Total_assets<-toString(fincondition3yearrmna[1,2])
			Total_debt<-toString(fincondition3yearrmna[1,3])
			Debt_ratio<-as.numeric(Total_debt)/as.numeric(Total_assets)
			Main_business_income_growth<-toString(profitincrease[1,2])
			Net_profit_growth<-toString(profitincrease[1,3])
			Net_asset_growth<-toString(profitincrease[1,4])
			Total_asset_growth<-toString(profitincrease[1,5])
			Profitgrowth_Pricegrowth_ratio_3year<-profitincreaseratio3y/StockPriceIncreaseRaito3year
			Profitgrowth_Pricegrowth_ratio_2year<-profitincreaseratio2y/StockPriceIncreaseRaito2year
			Profitgrowth_Pricegrowth_ratio_1year<-profitincreaseratio1y/StockPriceIncreaseRaito1year
			Profitgrowth_Pricegrowth_difference_3year<-profitincreaseratio3y-StockPriceIncreaseRaito3year
			Profitgrowth_Pricegrowth_difference_2year<-profitincreaseratio2y-StockPriceIncreaseRaito2year
			Profitgrowth_Pricegrowth_difference_1year<-profitincreaseratio1y-StockPriceIncreaseRaito1year

			Total_assets_profit_ratio<-toString(interestrate[1,2])
			The_main_business_profit_ratio<-toString(interestrate[1,3])
			Net_profit_ratio_of_total_assets<-toString(interestrate[1,4])
			Cost_profit<-toString(interestrate[1,5])
			Operating_profit_ratio<-toString(interestrate[1,6])
			The_main_business_cost_ratio<-toString(interestrate[1,7])
			Net_sales_ratio<-toString(interestrate[1,8])
			The_net_rate_of_return_on_assets<-toString(interestrate[1,9])
			Return_on_equity<-toString(interestrate[1,10])
			Return_on_net_assets<-toString(interestrate[1,11])
			Return_on_assets<-toString(interestrate[1,12])
			Sales_gross_margin<-toString(interestrate[1,13])
			Three_cost_proportion<-toString(interestrate[1,14])
			The_proportion_of_nonmain_business<-toString(interestrate[1,15])
			The_main_business_profit_proportion<-toString(interestrate[1,16])
#				profitincrease=paste(i,"cznl",sep="")[1:13,]
#				fincondition=paste(i,"zycwzb",sep="")[1:13,c(1:3,11:18,20)]
#			} else {
#				profitincrease=paste(i,"cznl",sep="")
#				fincondition=paste(i,"zycwzb",sep="")[,c(1:3,11:18,20)]
#			}
			finconditionYear=fincondition[substr(fincondition[,1],5,8)=="-12-",]
			#fincondition[grepl("-12-",fincondition[,1]),]
			profitincreaseYear=profitincrease[substr(profitincrease[,1],5,8)=="-12-",]
			interestrateYear=interestrate[substr(interestrate[,1],5,8)=="-12-",]

			fincondition5Years=head(finconditionYear, n=5)
			profitincrease5Years=head(profitincreaseYear, n=5)
			interestrate5Years=head(interestrateYear, n=5)
			price5Years = data.frame(tail(to.yearly(STOCKNAME), n=6))
			price5Yearssort = price5Years[rev(rownames(price5Years)), ]
			close5Years = price5Yearssort[substr(row.names(price5Yearssort),5,8)=="-12-",][,6]
			
			years = nrow(fincondition5Years)
			ROE_before = list()
			EPS_before = list()
			PE_before = list()
			price_before = list()
			cash_flow_before = list()
			cash_flow_per_share_before = list()
			Net_asset_growth_before = list()
			Total_asset_growth_before = list()
			Main_business_income_growth_before = list()
			Net_profit_growth_before = list()
			ROA_before = list()
			The_main_business_profit_ratio_before = list()
			The_main_business_profit_proportion_before = list()
			Net_sales_ratio_before = list()
			
			for (n in 1:years) {
				ROEy = paste("ROE_",n,sep="")
				ROE_before[ROEy] = fincondition5Years[,20][n]
				EPSid = paste("EPS_",n,sep="")
				EPSy = toString(fincondition5Years[,2][n])
				EPS_before[EPSid] = EPSy
				PEy = paste("PE_",n,sep="")
				PE_before[PEy] = as.numeric(close5Years[n])/as.numeric(EPSy)
				pricey = paste("price_",n,sep="")
				price_before[pricey] = close5Years[n]				
				cash_flowy = paste("cash_flow_",n,sep="")
				cash_flow_before[cash_flowy] = fincondition5Years[,13][n]
				cash_flow_per_sharey = paste("cash_flow_per_share_",n,sep="")
				cash_flow_per_share_before[cash_flow_per_sharey] = fincondition5Years[,4][n]
				Net_asset_growthy = paste("Net_asset_growth_",n,sep="")
				Net_asset_growth_before[Net_asset_growthy] = profitincrease5Years[,4][n]
				Total_asset_growthy = paste("Total_asset_growth_",n,sep="")
				Total_asset_growth_before[Total_asset_growthy] = profitincrease5Years[,5][n]
				Main_business_income_growthy = paste("Main_business_income_growth_",n,sep="")
				Main_business_income_growth_before[Main_business_income_growthy] = profitincrease5Years[,2][n]
				Net_profit_growthy = paste("Net_profit_growth_",n,sep="")
				Net_profit_growth_before[Net_profit_growthy] = profitincrease5Years[,3][n]
				ROAy = paste("ROA_",n,sep="")
				ROA_before[ROAy] = interestrate5Years[,4][n]
				The_main_business_profit_ratioy = paste("The_main_business_profit_ratio_",n,sep="")
				The_main_business_profit_ratio_before[The_main_business_profit_ratioy] = interestrate5Years[,3][n]
				The_main_business_profit_proportiony = paste("The_main_business_profit_proportion_",n,sep="")
				The_main_business_profit_proportion_before[The_main_business_profit_proportiony] = interestrate5Years[,16][n]
				Net_sales_ratioy = paste("Net_sales_ratio_",n,sep="")
				Net_sales_ratio_before[Net_sales_ratioy] = interestrate5Years[,8][n]									
			}
			if (all(na.omit(as.numeric(ROE_before)) == cummax(na.omit(as.numeric(ROE_before))))){
				ROE_increase = "ROE_decrease_every_year"
			} else if (all(na.omit(as.numeric(ROE_before)) == cummin(na.omit(as.numeric(ROE_before))))) {
				ROE_increase = "ROE_increase_every_year"
			} else {
				ROE_increase = "ROE_fluctuation"
			}
			if (all(na.omit(as.numeric(PE_before)) == cummax(na.omit(as.numeric(PE_before))))){
				PE_increase = "PE_decrease_every_year"
			} else if (all(na.omit(as.numeric(PE_before)) == cummin(na.omit(as.numeric(PE_before))))) {
				PE_increase = "PE_increase_every_year"
			} else {
				PE_increase = "PE_fluctuation"
			}
			
			# if (all(diff(as.numeric(PE_before))>= 0)){
				# PE_increase = "PE_decrease_every_year"
			# } else if (all(!diff(as.numeric(PE_before))>= 0)) {
				# PE_increase = "PE_increase_every_year"
			# } else {
				# PE_increase = "PE_fluctuation"
			# }
			# if (!is.unsorted(as.numeric(PE_before))){
				# PE_increase = "PE_decrease_every_year"
			# } else if (!is.unsorted(-as.numeric(PE_before))) {
				# PE_increase = "PE_increase_every_year"
			# } else {
				# PE_increase = "PE_fluctuation"
			# }
			if (all(na.omit(as.numeric(cash_flow_before)) == cummax(na.omit(as.numeric(cash_flow_before))))){
				cash_flow_increase = "cash_flow_decrease_every_year"
			} else if (all(na.omit(as.numeric(cash_flow_before)) == cummin(na.omit(as.numeric(cash_flow_before))))) {
				cash_flow_increase = "cash_flow_increase_every_year"
			} else {
				cash_flow_increase = "cash_flow_fluctuation"
			}
			if (all(na.omit(as.numeric(Net_profit_growth_before)) == cummax(na.omit(as.numeric(Net_profit_growth_before))))){
				Net_profit_growth_increase = "Net_profit_growth_decrease_every_year"
			} else if (all(na.omit(as.numeric(Net_profit_growth_before)) == cummin(na.omit(as.numeric(Net_profit_growth_before))))) {
				Net_profit_growth_increase = "Net_profit_growth_increase_every_year"
			} else {
				Net_profit_growth_increase = "Net_profit_growth_fluctuation"
			}
			if (all(na.omit(as.numeric(ROA_before)) == cummax(na.omit(as.numeric(ROA_before))))){
				ROA_increase = "ROA_decrease_every_year"
			} else if (all(na.omit(as.numeric(ROA_before)) == cummin(na.omit(as.numeric(ROA_before))))) {
				ROA_increase = "ROA_increase_every_year"
			} else {
				ROA_increase = "ROA_fluctuation"
			}
			The_main_business_profit_ratio_rmNA = as.numeric(The_main_business_profit_ratio_before)[!is.na(as.numeric(The_main_business_profit_ratio_before))]
			if (all(The_main_business_profit_ratio_rmNA == cummax(The_main_business_profit_ratio_rmNA))){
				The_main_business_profit_ratio_increase = "The_main_business_profit_ratio_decrease_every_year"
			} else if (all(The_main_business_profit_ratio_rmNA == cummin(The_main_business_profit_ratio_rmNA))) {
				The_main_business_profit_ratio_increase = "The_main_business_profit_ratio_increase_every_year"
			} else {
				The_main_business_profit_ratio_increase = "The_main_business_profit_ratio_fluctuation"
			}
			if (all(na.omit(as.numeric(Net_sales_ratio_before)) == cummax(na.omit(as.numeric(Net_sales_ratio_before))))){
				Net_sales_ratio_increase = "Net_sales_ratio_decrease_every_year"
			} else if (all(na.omit(as.numeric(Net_sales_ratio_before)) == cummin(na.omit(as.numeric(Net_sales_ratio_before))))) {
				Net_sales_ratio_increase = "Net_sales_ratio_increase_every_year"
			} else {
				Net_sales_ratio_increase = "Net_sales_ratio_fluctuation"
			}
			if (all(na.omit(as.numeric(fincondition5Years[,11])) == cummax(na.omit(as.numeric(fincondition5Years[,11]))))){
				Net_profile_increase = "Net_profile_decrease_every_year"
			} else if (all(na.omit(as.numeric(fincondition5Years[,11])) == cummin(na.omit(as.numeric(fincondition5Years[,11]))))) {
				Net_profile_increase = "Net_profile_increase_every_year"
			} else {
				Net_profile_increase = "Net_profile_fluctuation"
			}

			
			ROE5years_avg = mean(as.numeric(ROE_before))
			ROE5years_sd = sd(as.numeric(ROE_before))
			PE5years_avg = mean(as.numeric(PE_before))
			PE5years_sd = sd(as.numeric(PE_before))
			price5years_avg = mean(as.numeric(price_before))
			price5years_sd = sd(as.numeric(price_before))
			cash_flow_avg = mean(as.numeric(cash_flow_before))
			cash_flow_sd = sd(as.numeric(cash_flow_before))
			Net_asset_growth5years_avg = mean(as.numeric(Net_asset_growth_before))
			Total_asset_growth5years_avg = mean(as.numeric(Total_asset_growth_before))
			Main_business_income_growth5years_avg = mean(as.numeric(Main_business_income_growth_before))
			Net_profit_growth5years_avg = mean(as.numeric(Net_profit_growth_before))
			Net_asset_growth5years_sd = sd(as.numeric(Net_asset_growth_before))
			Total_asset_growth5years_sd = sd(as.numeric(Total_asset_growth_before))
			Main_business_income_growth5years_sd = sd(as.numeric(Main_business_income_growth_before))
			Net_profit_growth5years_sd = sd(as.numeric(Net_profit_growth_before))
			ROA5years_avg = mean(as.numeric(ROA_before))
			ROA5years_sd = sd(as.numeric(ROA_before))
			The_main_business_profit_ratio5years_avg = mean(as.numeric(The_main_business_profit_ratio_before))
			The_main_business_profit_ratio5years_sd = sd(as.numeric(The_main_business_profit_ratio_before))
			The_main_business_profit_proportion5years_avg = mean(as.numeric(The_main_business_profit_proportion_before))
			The_main_business_profit_proportion5years_sd = sd(as.numeric(The_main_business_profit_proportion_before))
			Net_sales_ratio5years_avg = mean(as.numeric(Net_sales_ratio_before))
			Net_sales_ratio5years_sd = sd(as.numeric(Net_sales_ratio_before))
			
			
			
			
		mat<-as.matrix(matrix(c(i,Main_business_income_growth,Net_asset_growth,Total_asset_growth,Net_profit_growth,Profitgrowth_Pricegrowth_ratio_3year,Profitgrowth_Pricegrowth_difference_3year,StockPriceIncreaseRaito3year,profitincreaseratio3y,profitgrowth_increase,Open,High,Low,Close,Volume,Adjusted,EPS,PE,PB,ROE,Net_profit,cash_flow,Current_assets,Total_assets,Total_debt,Debt_ratio,Total_assets_profit_ratio,The_main_business_profit_ratio,Net_profit_ratio_of_total_assets,Cost_profit,Operating_profit_ratio,The_main_business_cost_ratio,Net_sales_ratio,Return_on_equity,Return_on_net_assets,Return_on_assets,Sales_gross_margin,Three_cost_proportion,The_proportion_of_nonmain_business,The_main_business_profit_proportion,Profitgrowth_Pricegrowth_ratio_2year,Profitgrowth_Pricegrowth_difference_2year,profitincreaseratio2y,StockPriceIncreaseRaito2year,Profitgrowth_Pricegrowth_ratio_1year,Profitgrowth_Pricegrowth_difference_1year,profitincreaseratio1y,StockPriceIncreaseRaito1year,StockPriceIncreaseRaito6month,StockPriceIncreaseRaito3month,StockPriceIncreaseRaito1month,ROE_before,PE_before,price_before,cash_flow_before,cash_flow_per_share_before,Net_asset_growth_before,Total_asset_growth_before,Main_business_income_growth_before,Net_profit_growth_before,ROA_before,The_main_business_profit_ratio_before,The_main_business_profit_proportion_before,Net_sales_ratio_before,ROE_increase,PE_increase,cash_flow_increase,Net_profit_growth_increase,ROA_increase,The_main_business_profit_ratio_increase,Net_sales_ratio_increase,Net_profile_increase,ROE5years_avg,ROE5years_sd,PE5years_avg,PE5years_sd,price5years_avg,price5years_sd,cash_flow_avg,cash_flow_sd,Net_asset_growth5years_avg,Net_asset_growth5years_sd,Total_asset_growth5years_avg,Total_asset_growth5years_sd,Main_business_income_growth5years_avg,Main_business_income_growth5years_sd,Net_profit_growth5years_avg,Net_profit_growth5years_sd,ROA5years_avg,ROA5years_sd,The_main_business_profit_ratio5years_avg,The_main_business_profit_ratio5years_sd,The_main_business_profit_proportion5years_avg,The_main_business_profit_proportion5years_sd,Net_sales_ratio5years_avg,Net_sales_ratio5years_sd),nrow=1))
		colnames(mat)<-c("Codes","Main_business_income_growth","Net_asset_growth","Total_asset_growth","Net_profit_growth","Profitgrowth_Pricegrowth_ratio_3year","Profitgrowth_Pricegrowth_difference_3year","StockPrice_growth_rate_last3years","Profit_GrowthRate_last3year","profitgrowth_increase","Open","High","Low","Close","Volume","Adjusted","EPS","PE","PB","ROE","Net_profit","cash_flow","Current_assets","Total_assets","Total_debt","Debt_ratio","Total_assets_profit_ratio","The_main_business_profit_ratio","Net_profit_ratio_of_total_assets","Cost_profit","Operating_profit_ratio","The_main_business_cost_ratio","Net_sales_ratio","Return_on_equity","Return_on_net_assets","Return_on_assets","Sales_gross_margin","Three_cost_proportion","The_proportion_of_nonmain_business","The_main_business_profit_proportion","Profitgrowth_Pricegrowth_ratio_2year","Profitgrowth_Pricegrowth_difference_2year","profitincreaseratio2y","StockPriceIncreaseRaito2year","Profitgrowth_Pricegrowth_ratio_1year","Profitgrowth_Pricegrowth_difference_1year","profitincreaseratio1y","StockPriceIncreaseRaito1year","StockPriceIncreaseRaito6month","StockPriceIncreaseRaito3month","StockPriceIncreaseRaito1month","ROE_before_1ya","ROE_before_2ya","ROE_before_3ya","ROE_before_4ya","ROE_before_5ya","PE_before_1ya","PE_before_2ya","PE_before_3ya","PE_before_4ya","PE_before_5ya","price_before_1ya","price_before_2ya","price_before_3ya","price_before_4ya","price_before_5ya","cash_flow_before_1ya","cash_flow_before_2ya","cash_flow_before_3ya","cash_flow_before_4ya","cash_flow_before_5ya","cash_flow_per_share_before_1ya","cash_flow_per_share_before_2ya","cash_flow_per_share_before_3ya","cash_flow_per_share_before_4ya","cash_flow_per_share_before_5ya","Net_asset_growth_before_1ya","Net_asset_growth_before_2ya","Net_asset_growth_before_3ya","Net_asset_growth_before_4ya","Net_asset_growth_before_5ya","Total_asset_growth_before_1ya","Total_asset_growth_before_2ya","Total_asset_growth_before_3ya","Total_asset_growth_before_4ya","Total_asset_growth_before_5ya","Main_business_income_growth_before_1ya","Main_business_income_growth_before_2ya","Main_business_income_growth_before_3ya","Main_business_income_growth_before_4ya","Main_business_income_growth_before_5ya","Net_profit_growth_before_1ya","Net_profit_growth_before_2ya","Net_profit_growth_before_3ya","Net_profit_growth_before_4ya","Net_profit_growth_before_5ya","ROA_before_1ya","ROA_before_2ya","ROA_before_3ya","ROA_before_4ya","ROA_before_5ya","The_main_business_profit_ratio_before_1ya","The_main_business_profit_ratio_before_2ya","The_main_business_profit_ratio_before_3ya","The_main_business_profit_ratio_before_4ya","The_main_business_profit_ratio_before_5ya","The_main_business_profit_proportion_before_1ya","The_main_business_profit_proportion_before_2ya","The_main_business_profit_proportion_before_3ya","The_main_business_profit_proportion_before_4ya","The_main_business_profit_proportion_before_5ya","Net_sales_ratio_before_1ya","Net_sales_ratio_before_2ya","Net_sales_ratio_before_3ya","Net_sales_ratio_before_4ya","Net_sales_ratio_before_5ya","ROE_increase","PE_increase","cash_flow_increase","Net_profit_growth_increase","ROA_increase","The_main_business_profit_ratio_increase","Net_sales_ratio_increase","Net_profile_increase","ROE5years_avg","ROE5years_sd","PE5years_avg","PE5years_sd","price5years_avg","price5years_sd","cash_flow_avg","cash_flow_sd","Net_asset_growth5years_avg","Net_asset_growth5years_sd","Total_asset_growth5years_avg","Total_asset_growth5years_sd","Main_business_income_growth5years_avg","Main_business_income_growth5years_sd","Net_profit_growth5years_avg","Net_profit_growth5years_sd","ROA5years_avg","ROA5years_sd","The_main_business_profit_ratio5years_avg","The_main_business_profit_ratio5years_sd","The_main_business_profit_proportion5years_avg","The_main_business_profit_proportion5years_sd","Net_sales_ratio5years_avg","Net_sales_ratio5years_sd")

		#		colnames(matlist[i,])<-c("Stock_Codes","Main_business_income_growth","Net_asset_growth","Total_asset_growth","Net_profit_growth","Profitgrowth_Pricegrowth_ratio_3year","StockPrice_growth_rate_last3years","Profit_GrowthRate_last3year","Open","High","Low","Close","Volume","Adjusted","EPS","PE","PB","ROE","Net_profit","cash_flow","Current_assets","Total_assets","Total_debt","Debt_ratio")

#		mat<-matlist[stocksA[,"Codes"][1],]
#		for (i in stocksA[,"Codes"][2]) {
#		mat=rbind(mat,matlist[i,])
#		mergedata = do.call("rbind",as.name(paste("Stock",i,sep="")))
		filename<-paste(i,".csv",sep="")
		write.csv(mat,filename,row.names=F,col.names=T)

#		}
  
	
}
stocksA1<-read.table("A3.txt",header=T,encoding="UTF-8")
filelist <- list.files(pattern=".*.csv")
datalist <- lapply(filelist, function(x) read.csv(x, header=T,sep=",")) 
datafr <- do.call("rbind", datalist)
AplusName<- merge(stocksA1,datafr,by="Codes",all=T)
AplusName1<-AplusName[rowSums(is.na(AplusName[,3:52]))!=50,]
AplusName1a<-AplusName1[complete.cases(AplusName1[1:2]),]
write.csv(AplusName1a,"A_stocks_assessment.csv",row.names=F,col.names=T)