# project-14 與 project-16 標註差異

## 比對範圍
- 舊標註：`data/origin_data/10k_1A/project-14-at-2026-04-06-06-12-80d88013.csv`
- 新標註：`data/origin_data/10k_1A/project-16-at-2026-03-31-11-52-a8826fd3.csv`
- 對齊鍵：`paragraph_id`
- 比對結果：兩檔皆為 500 筆，`paragraph_id` 完全一致
- 文字欄位檢查：`combined_text`、`risk_heading`、`ticker`、`filing_date`、`finbert_label`、`finbert_confidence` 皆無差異，實際變動只有標註結果與標註 metadata

## 摘要
- 總樣本數：500
- 標註有變動：78 筆（15.6%）
- `Non-ESG -> ESG`：42 筆
- `ESG -> Non-ESG`：24 筆
- `ESG -> ESG`：12 筆
- `(空白) -> Non-ESG`：1 筆

## 標籤轉換統計
| 舊標籤 | 新標籤 | 筆數 |
| --- | --- | ---: |
| Non-ESG | Corporate Governance | 9 |
| Product Liability | Non-ESG | 9 |
| Non-ESG | Human Capital | 8 |
| Non-ESG | Product Liability | 8 |
| Non-ESG | Business Ethics & Values | 7 |
| Non-ESG | Climate Change | 5 |
| Corporate Governance | Non-ESG | 5 |
| Business Ethics & Values | Non-ESG | 4 |
| Human Capital | Non-ESG | 4 |
| Non-ESG | Community Relations | 3 |
| Pollution & Waste | Climate Change | 2 |
| Community Relations | Product Liability | 1 |
| Non-ESG | Natural Capital | 1 |
| Product Liability | Community Relations | 1 |
| Non-ESG | Pollution & Waste | 1 |
| Pollution & Waste | Human Capital | 1 |
| Product Liability | Natural Capital | 1 |
| Product Liability | Climate Change | 1 |
| Business Ethics & Values | Product Liability | 1 |
| Product Liability | Corporate Governance | 1 |
| Product Liability | Business Ethics & Values | 1 |
| Climate Change | Non-ESG | 1 |
| Natural Capital | Community Relations | 1 |
| (空白) | Non-ESG | 1 |
| Product Liability | Human Capital | 1 |

## 明細
| # | paragraph_id | ticker | filing_date | 舊標籤 | 新標籤 | FinBERT | 信心 | risk_heading | 文字摘錄 |
| ---: | --- | --- | --- | --- | --- | --- | ---: | --- | --- |
| 1 | AKAM_2022_RISK_059 | AKAM | 2022-02-28 | Non-ESG | Corporate Governance | Corporate Governance | 0.4512 | Legal and Regulatory Risks | Legal and Regulatory Risks Litigation may adversely impact our business. From time to time, we are or may b... |
| 2 | ALGN_2022_RISK_045 | ALGN | 2022-02-25 | Non-ESG | Business Ethics & Values | Business Ethics & Values | 0.5774 | Risks Relating to our Business Operations and Strategy | Risks Relating to our Business Operations and Strategy Moreover, we have developed a multi-dimensional busi... |
| 3 | AVY_2021_RISK_038 | AVY | 2021-02-25 | Non-ESG | Business Ethics & Values | Non-ESG | 0.8479 | Risks Related to COVID-19 | Risks Related to COVID-19 We attempt to protect and restrict access to our intellectual property and propri... |
| 4 | A_2021_RISK_061 | A | 2021-12-17 | Non-ESG | Climate Change | Non-ESG | 0.7239 | Operational Risks | Operational Risks Our factories, facilities and distribution system are subject to catastrophic loss due to... |
| 5 | BBY_2021_RISK_011 | BBY | 2021-03-19 | Community Relations | Product Liability | Product Liability | 0.8457 | The COVID-19 pandemic has subjected our business, operations and financial condition to a number of risks, and those risks may intensify or last for an extended period of time. | The COVID-19 pandemic has subjected our business, operations and financial condition to a number of risks, ... |
| 6 | BEN_2021_RISK_075 | BEN | 2021-11-19 | Non-ESG | Corporate Governance | Corporate Governance | 0.9932 | INFORMATION ABOUT OUR EXECUTIVE OFFICERS | INFORMATION ABOUT OUR EXECUTIVE OFFICERS The following description of our executive officers is included as... |
| 7 | BKNG_2022_RISK_079 | BKNG | 2022-02-23 | Business Ethics & Values | Non-ESG | Non-ESG | 0.4649 | Legal, Tax, Regulatory, Compliance, and Reputational Risks | Legal, Tax, Regulatory, Compliance, and Reputational Risks Some parts of our business are already subject t... |
| 8 | CAG_2021_RISK_049 | CAG | 2021-07-23 | Product Liability | Non-ESG | Non-ESG | 0.8351 | Cybersecurity and Information Technology Risks | Cybersecurity and Information Technology Risks Our business operations could be disrupted if our informatio... |
| 9 | CARR_2022_RISK_021 | CARR | 2022-02-08 | Non-ESG | Corporate Governance | Corporate Governance | 0.8777 | Risks Related to Our Business | Risks Related to Our Business Whether or not we hold a majority interest or maintain operational control in... |
| 10 | CDNS_2021_RISK_030 | CDNS | 2021-02-22 | Non-ESG | Natural Capital | Non-ESG | 0.4753 | Our business is subject to the risk of earthquakes and other catastrophic events. | Our business is subject to the risk of earthquakes and other catastrophic events. Our corporate headquarter... |
| 11 | CE_2022_RISK_061 | CE | 2022-02-10 | Human Capital | Non-ESG | Non-ESG | 0.9717 | Risks Related to Our Human Capital | Risks Related to Our Human Capital Significant changes in pension fund investment performance or assumption... |
| 12 | CHD_2022_RISK_051 | CHD | 2022-02-17 | Product Liability | Community Relations | Product Liability | 0.9339 | • We rely significantly on information technology. Any inadequacy, interruption, theft or loss of data, malicious attack, integration failure, failure to maintain the security, confidentiality or privacy of sensitive data residing on our systems or other security failure of that technology could harm our ability to effectively operate our business and damage the reputation of our brands. | • We rely significantly on information technology. Any inadequacy, interruption, theft or loss of data, mal... |
| 13 | CME_2022_RISK_020 | CME | 2022-02-25 | Business Ethics & Values | Non-ESG | Non-ESG | 0.4437 | RISKS RELATING TO OUR INDUSTRY | RISKS RELATING TO OUR INDUSTRY Many aspects of our business present substantial litigation risks. These ris... |
| 14 | CMG_2022_RISK_034 | CMG | 2022-02-11 | Non-ESG | Human Capital | Human Capital | 0.8246 | Risks Related to Human Capital | Risks Related to Human Capital Our quarterly financial results may fluctuate significantly and could fail t... |
| 15 | CNC_2023_RISK_016 | CNC | 2023-02-21 | Product Liability | Non-ESG | Non-ESG | 0.7603 | Risks Relating to Our Business | Risks Relating to Our Business A substantial portion of our business relates to the provision of managed ca... |
| 16 | COR_2021_RISK_050 | COR | 2021-11-23 | Product Liability | Non-ESG | Non-ESG | 0.5769 | Litigation and Regulatory Risks | Litigation and Regulatory Risks We recorded a charge of $6.6 billion in the fiscal year ended September 30,... |
| 17 | CRL_2023_RISK_018 | CRL | 2023-02-22 | Non-ESG | Corporate Governance | Non-ESG | 0.6383 | Business and Operational Risks | Business and Operational Risks Acquisitions and alliances involve numerous risks which may include:•difficu... |
| 18 | CVX_2022_RISK_003 | CVX | 2022-02-24 | Non-ESG | Pollution & Waste | Non-ESG | 0.8629 | BUSINESS AND OPERATIONAL RISK FACTORS | BUSINESS AND OPERATIONAL RISK FACTORS The single largest variable that affects the company’s results of ope... |
| 19 | DE_2021_RISK_002 | DE | 2021-12-16 | Non-ESG | Community Relations | Community Relations | 0.3257 | Risks Related to the COVID Pandemic | Risks Related to the COVID Pandemic The virus causing COVID was identified in late 2019 and spread globally... |
| 20 | DOC_2022_RISK_038 | DOC | 2022-02-09 | Non-ESG | Corporate Governance | Corporate Governance | 0.5028 | If we are unable to successfully integrate our acquisitions, our business, results of operations and financial condition may be materially adversely affected. | If we are unable to successfully integrate our acquisitions, our business, results of operations and financ... |
| 21 | DRI_2021_RISK_059 | DRI | 2021-07-23 | Human Capital | Non-ESG | Non-ESG | 0.8949 | Risks Relating to Our Business Model and Strategy | Risks Relating to Our Business Model and Strategy The equity markets in the U.S. were extremely volatile du... |
| 22 | DVA_2022_RISK_029 | DVA | 2022-02-11 | Product Liability | Non-ESG | Non-ESG | 0.8973 | If the number or percentage of patients with higher-paying commercial insurance declines, if the average rates that commercial payors pay us decline, if patients in commercial plans are subject to restriction in plan designs, if we are unable to maintain contracts with payors with competitive terms, including, without limitation, reimbursement rates, scope and duration of coverage and in-network benefits, it could have a material adverse effect on our business, results of operations, financial condition and cash flows. | If the number or percentage of patients with higher-paying commercial insurance declines, if the average ra... |
| 23 | DVN_2022_RISK_007 | DVN | 2022-02-16 | Non-ESG | Human Capital | Human Capital | 0.4913 | We Are Subject to Extensive Governmental Regulation, Which Can Change and Could Adversely Impact Our Business | We Are Subject to Extensive Governmental Regulation, Which Can Change and Could Adversely Impact Our Busine... |
| 24 | D_2022_RISK_020 | D | 2022-02-24 | Pollution & Waste | Human Capital | Human Capital | 0.7193 | Operational Risks | Operational Risks Unplanned outages of the Companies’ facilities and extensions of scheduled outages due to... |
| 25 | EG_2022_RISK_015 | EG | 2022-02-28 | Non-ESG | Human Capital | Human Capital | 0.9693 | RISKS RELATING TO OUR BUSINESS | RISKS RELATING TO OUR BUSINESS engage in any gainful occupation in Bermuda without a work permit issued by ... |
| 26 | ENPH_2023_RISK_157 | ENPH | 2023-02-13 | Non-ESG | Climate Change | Non-ESG | 0.4521 | General Risks Related to our Business | General Risks Related to our Business Our worldwide operations could be subject to natural disasters (inclu... |
| 27 | EVRG_2022_RISK_008 | EVRG | 2022-02-25 | Pollution & Waste | Climate Change | Climate Change | 0.6152 | Environmental Risks: | Environmental Risks: The new interpretations could require modified compliance plans such as different meth... |
| 28 | EXPD_2022_RISK_016 | EXPD | 2022-03-15 | Business Ethics & Values | Non-ESG | Non-ESG | 0.7879 | We Face Risks Associated with the Handling of Customer Inventory | We Face Risks Associated with the Handling of Customer Inventory As a multinational corporation, Expeditors... |
| 29 | EXPE_2026_RISK_004 | EXPE | 2026-02-13 | Product Liability | Natural Capital | Corporate Governance | 0.6487 | Cybersecurity Risk Management and Strategy | Cybersecurity Risk Management and Strategy The Company’s mandatory annual cybersecurity employee training p... |
| 30 | FANG_2023_RISK_084 | FANG | 2023-02-23 | Non-ESG | Corporate Governance | Corporate Governance | 0.748 | Risks Related to the Oil and Natural Gas Industry and Our Business | Risks Related to the Oil and Natural Gas Industry and Our Business From time to time, legislation has been ... |
| 31 | FITB_2022_RISK_007 | FITB | 2022-02-25 | Non-ESG | Business Ethics & Values | Business Ethics & Values | 0.4385 | REPUTATION RISKS | REPUTATION RISKS •Damage to Fifth Third’s reputation could harm its business.•Fifth Third is subject to env... |
| 32 | FITB_2023_RISK_031 | FITB | 2023-02-24 | Product Liability | Climate Change | Product Liability | 0.8026 | OPERATIONAL RISKS | OPERATIONAL RISKS Fifth Third’s operations, including its financial and accounting systems, use computer sy... |
| 33 | GD_2022_RISK_012 | GD | 2022-02-09 | Non-ESG | Human Capital | Human Capital | 0.737 | Other Business and Operational Risks | Other Business and Operational Risks However, future threats could, among other things, cause harm to our b... |
| 34 | GILD_2022_RISK_023 | GILD | 2022-02-23 | Business Ethics & Values | Product Liability | Non-ESG | 0.4858 | Our operations depend on compliance with complex FDA and comparable international regulations. Failure to obtain broad approvals on a timely basis or to maintain compliance could delay or halt commercialization of our products. | Our operations depend on compliance with complex FDA and comparable international regulations. Failure to o... |
| 35 | HLT_2023_RISK_006 | HLT | 2023-02-09 | Non-ESG | Climate Change | Climate Change | 0.3268 | Risks Related to Our Industry | Risks Related to Our Industry Federal government shutdowns and other similar governmental budgetary impasse... |
| 36 | IFF_2022_RISK_006 | IFF | 2022-02-28 | Non-ESG | Product Liability | Product Liability | 0.4464 | Supply chain disruptions, geopolitical developments or climate-change events may adversely affect our suppliers or our procurement of raw materials, and thus may impact our business and financial results. | Supply chain disruptions, geopolitical developments or climate-change events may adversely affect our suppl... |
| 37 | INVH_2024_RISK_135 | INVH | 2024-02-21 | Non-ESG | Corporate Governance | Corporate Governance | 0.5502 | Risks Related to our REIT Status and Certain Other Tax Items | Risks Related to our REIT Status and Certain Other Tax Items We believe that we have been organized and hav... |
| 38 | IRM_2022_RISK_055 | IRM | 2022-02-24 | Corporate Governance | Non-ESG | Non-ESG | 0.6011 | RISKS RELATED TO OUR INDEBTEDNESS | RISKS RELATED TO OUR INDEBTEDNESS Upon the occurrence of a “change of control,” as defined in our indenture... |
| 39 | ISRG_2022_RISK_045 | ISRG | 2022-02-03 | Non-ESG | Product Liability | Non-ESG | 0.6792 | WE UTILIZE DISTRIBUTORS FOR A PORTION OF OUR SALES AND SERVICE OF OUR PRODUCTS IN CERTAIN COUNTRIES, WHICH SUBJECTS US TO A NUMBER OF RISKS THAT COULD HARM OUR BUSINESS. | WE UTILIZE DISTRIBUTORS FOR A PORTION OF OUR SALES AND SERVICE OF OUR PRODUCTS IN CERTAIN COUNTRIES, WHICH ... |
| 40 | KDP_2022_RISK_013 | KDP | 2022-02-24 | Non-ESG | Product Liability | Product Liability | 0.7387 | RISKS RELATED TO OUR OPERATIONS | RISKS RELATED TO OUR OPERATIONS Consumers’ preferences continually evolve due to a variety of factors, incl... |
| 41 | KEY_2022_RISK_013 | KEY | 2022-02-22 | Non-ESG | Business Ethics & Values | Business Ethics & Values | 0.5053 | We are subject to a variety of operational risks. | We are subject to a variety of operational risks. In addition to the other risks discussed in this section,... |
| 42 | KLAC_2021_RISK_024 | KLAC | 2021-08-06 | Pollution & Waste | Climate Change | Climate Change | 0.7717 | Commercial, Operational, Financial and Regulatory Risks | Commercial, Operational, Financial and Regulatory Risks Our properties and many aspects of our business ope... |
| 43 | LW_2021_RISK_002 | LW | 2021-07-27 | Non-ESG | Community Relations | Community Relations | 0.6401 | Business and Operating Risks | Business and Operating Risks The ultimate impact that the COVID-19 pandemic and any future pandemic or othe... |
| 44 | MCHP_2022_RISK_061 | MCHP | 2022-05-20 | Product Liability | Corporate Governance | Corporate Governance | 0.4908 | Risks Related to Cybersecurity, Privacy, Intellectual Property, and Litigation | Risks Related to Cybersecurity, Privacy, Intellectual Property, and Litigation We do not believe that this ... |
| 45 | MCK_2021_RISK_049 | MCK | 2021-05-12 | Product Liability | Non-ESG | Non-ESG | 0.8486 | Industry and Economic Risks | Industry and Economic Risks We may have difficulties in sourcing or selling products due to a variety of ca... |
| 46 | MOH_2024_RISK_012 | MOH | 2024-02-13 | Corporate Governance | Non-ESG | Non-ESG | 0.9702 | RISKS RELATED TO OUR BUSINESS | RISKS RELATED TO OUR BUSINESS Our growth strategy includes the pursuit of targeted inorganic growth opportu... |
| 47 | MSCI_2022_RISK_080 | MSCI | 2022-02-11 | Corporate Governance | Non-ESG | Non-ESG | 0.9487 | General Risks | General Risks We cannot provide any guaranty that we will continue to repurchase shares of our common stock... |
| 48 | NCLH_2022_RISK_036 | NCLH | 2022-03-01 | Non-ESG | Human Capital | Human Capital | 0.8916 | COVID-19 and Debt/Liquidity Related Risk Factors | COVID-19 and Debt/Liquidity Related Risk Factors We have also made, and plan to continue to make, investmen... |
| 49 | NDAQ_2022_RISK_043 | NDAQ | 2022-02-23 | Non-ESG | Product Liability | Product Liability | 0.4779 | RISKS RELATED TO OUR BUSINESS AND INDUSTRY | RISKS RELATED TO OUR BUSINESS AND INDUSTRY Some locations, such as Lithuania, India and the Philippines, ha... |
| 50 | NEM_2023_RISK_102 | NEM | 2023-02-23 | Non-ESG | Community Relations | Community Relations | 0.7161 | Risks Related to the Jurisdictions in Which We Operate | Risks Related to the Jurisdictions in Which We Operate Our Merian operation in Suriname is subject to polit... |
| 51 | NVDA_2021_RISK_006 | NVDA | 2021-02-26 | Product Liability | Non-ESG | Non-ESG | 0.9664 | We depend on third parties and their technology to manufacture, assemble, test and/or package our products, which reduces our control over product quantity and quality, manufacturing yields, development, enhancement and product delivery schedule and could harm our business. | We depend on third parties and their technology to manufacture, assemble, test and/or package our products,... |
| 52 | PAYX_2021_RISK_017 | PAYX | 2021-07-16 | Business Ethics & Values | Non-ESG | Non-ESG | 0.7268 | Business and Operational Risks | Business and Operational Risks We may also be obligated to indemnify our customers or vendors in connection... |
| 53 | PCG_2022_RISK_017 | PCG | 2022-02-10 | Non-ESG | Human Capital | Human Capital | 0.9702 | Risks Related to Operations and Information Technology | Risks Related to Operations and Information Technology The Utility’s ability to efficiently construct, main... |
| 54 | PFE_2022_RISK_008 | PFE | 2022-02-24 | Product Liability | Non-ESG | Non-ESG | 0.475 | RESEARCH AND DEVELOPMENT | RESEARCH AND DEVELOPMENT R&D is at the heart of fulfilling our purpose to deliver breakthroughs that change... |
| 55 | PFG_2022_RISK_033 | PFG | 2022-02-11 | Product Liability | Business Ethics & Values | Business Ethics & Values | 0.6801 | Risks relating to estimates, assumptions and valuations | Risks relating to estimates, assumptions and valuations It requires financial professionals to act in consu... |
| 56 | PKG_2022_RISK_009 | PKG | 2022-02-24 | Non-ESG | Product Liability | Product Liability | 0.6413 | Risks Related to our Operations, Business and Industry | Risks Related to our Operations, Business and Industry •Explosion of a boiler or other major facilities.•Di... |
| 57 | POOL_2022_RISK_028 | POOL | 2022-02-25 | Non-ESG | Business Ethics & Values | Business Ethics & Values | 0.6691 | Risks Relating to Legal, Regulatory and Compliance Matters | Risks Relating to Legal, Regulatory and Compliance Matters The nature of our business subjects us to compli... |
| 58 | PPL_2022_RISK_019 | PPL | 2022-02-18 | Climate Change | Non-ESG | Non-ESG | 0.5498 | D. Risks Specific to Pennsylvania Regulated Segment | D. Risks Specific to Pennsylvania Regulated Segment PPL Electric is subject to Act 129, which contains requ... |
| 59 | PRU_2022_RISK_043 | PRU | 2022-02-17 | Human Capital | Non-ESG | Non-ESG | 0.8184 | Strategic Risk | Strategic Risk •We may not realize or sustain the expected benefits from programs we have announced, and th... |
| 60 | PSA_2022_RISK_035 | PSA | 2022-02-22 | Non-ESG | Climate Change | Climate Change | 0.9925 | Properties | Properties (a)See Schedule III: Real Estate and Accumulated Depreciation in our consolidated financial stat... |
| 61 | PWR_2023_RISK_074 | PWR | 2023-02-23 | Non-ESG | Climate Change | Climate Change | 0.5002 | Risks Related to Operating Our Business | Risks Related to Operating Our Business Pursuant to certain contracts, including fixed price and EPC contra... |
| 62 | PYPL_2022_RISK_028 | PYPL | 2022-02-03 | Non-ESG | Product Liability | Non-ESG | 0.9658 | Privacy and Protection of Customer Data | Privacy and Protection of Customer Data matters involves judgment and may not reflect the full range of unc... |
| 63 | REG_2022_RISK_044 | REG | 2022-02-17 | Non-ESG | Corporate Governance | Corporate Governance | 0.9612 | Dividends paid by REITs generally do not qualify for reduced tax rates. | Dividends paid by REITs generally do not qualify for reduced tax rates. Subject to limited exceptions, divi... |
| 64 | ROST_2021_RISK_010 | ROST | 2021-03-30 | Non-ESG | Human Capital | Non-ESG | 0.9607 | We depend on the market availability, quantity, and quality of attractive brand name merchandise at desirable discounts, and on the ability of our buyers to purchase merchandise to enable us to offer customers a wide assortment of merchandise at competitive prices. | We depend on the market availability, quantity, and quality of attractive brand name merchandise at desirab... |
| 65 | SCHW_2022_RISK_036 | SCHW | 2022-02-24 | Corporate Governance | Non-ESG | Non-ESG | 0.6134 | Future sales of CSC’s equity securities may adversely affect the market price of CSC’s common stock and result in dilution. | Future sales of CSC’s equity securities may adversely affect the market price of CSC’s common stock and res... |
| 66 | SHW_2022_RISK_053 | SHW | 2022-02-17 | Product Liability | Non-ESG | Non-ESG | 0.8207 | LEGAL AND REGULATORY RISKS | LEGAL AND REGULATORY RISKS Notwithstanding our views on the merits, litigation is inherently subject to man... |
| 67 | STZ_2021_RISK_008 | STZ | 2021-04-20 | Natural Capital | Community Relations | Natural Capital | 0.7674 | Dependence on limited facilities for production of our Mexican beer brands, and expansion and construction issues | Dependence on limited facilities for production of our Mexican beer brands, and expansion and construction ... |
| 68 | SWK_2021_RISK_058 | SWK | 2021-02-18 | (空白) | Non-ESG | Non-ESG | 0.9923 | Note H, Long-Term Debt and Financing Arrangements | Note H, Long-Term Debt and Financing Arrangements Discontinuation, reform or replacement of the London Inte... |
| 69 | TRGP_2022_RISK_047 | TRGP | 2022-02-24 | Non-ESG | Business Ethics & Values | Business Ethics & Values | 0.5529 | Risks Related to our Results of Operations | Risks Related to our Results of Operations We have experienced, and we anticipate that we will encounter fr... |
| 70 | TRV_2022_RISK_013 | TRV | 2022-02-17 | Non-ESG | Human Capital | Human Capital | 0.9568 | Insurance-Related Risks | Insurance-Related Risks Examples of such claims and coverage issues include, but are not limited to:•judici... |
| 71 | TTWO_2022_RISK_082 | TTWO | 2022-05-17 | Non-ESG | Corporate Governance | Non-ESG | 0.4906 | Risks relating to our business and industry | Risks relating to our business and industry We develop proprietary software and have obtained the rights to... |
| 72 | TT_2022_RISK_022 | TT | 2022-02-07 | Product Liability | Non-ESG | Non-ESG | 0.7327 | Risks Related to Cybersecurity and Technology | Risks Related to Cybersecurity and Technology We are subject to risks relating to our information technolog... |
| 73 | UAL_2022_RISK_056 | UAL | 2022-02-18 | Non-ESG | Business Ethics & Values | Business Ethics & Values | 0.4511 | Regulatory, Tax, Litigation and Legal Compliance Risks | Regulatory, Tax, Litigation and Legal Compliance Risks From time to time, we are subject to litigation and ... |
| 74 | UNH_2022_RISK_043 | UNH | 2022-02-15 | Corporate Governance | Non-ESG | Non-ESG | 0.9783 | Restrictions on our ability to obtain funds from our regulated subsidiaries could materially and adversely affect our results of operations, financial position and cash flows. | Restrictions on our ability to obtain funds from our regulated subsidiaries could materially and adversely ... |
| 75 | VRSN_2022_RISK_034 | VRSN | 2022-02-18 | Non-ESG | Product Liability | Non-ESG | 0.9691 | The evolution of technologies or internet practices and behaviors, the adoption of substitute technologies, or wholesale price increases of domain names in our TLDs may materially and negatively impact the demand for the domain names for which we are the registry operator. | The evolution of technologies or internet practices and behaviors, the adoption of substitute technologies,... |
| 76 | VST_2023_RISK_116 | VST | 2023-03-01 | Human Capital | Non-ESG | Non-ESG | 0.9295 | Operational Risks | Operational Risks The loss of the services of our key management and personnel could adversely affect our a... |
| 77 | XOM_2022_RISK_006 | XOM | 2022-02-23 | Non-ESG | Product Liability | Product Liability | 0.5842 | Government and Political Factors | Government and Political Factors We also may be adversely affected by the outcome of litigation, especially... |
| 78 | YUM_2022_RISK_009 | YUM | 2022-02-23 | Product Liability | Human Capital | Human Capital | 0.4202 | Risks Related to COVID-19, Food Safety and Catastrophic Events | Risks Related to COVID-19, Food Safety and Catastrophic Events Our business could be materially and adverse... |
