#!/usr/bin/env python
#-*- coding: utf-8 -*-
# ===================================================================================================================
#  Author:       v_sxsun
#  CreateDate:   2020-12-15
#  FileName:     t_app_qb_cidnum_real_exp_sum_d.py
#  Riviewer:     ivanqu
#  GroupID:      g_pcg_txkd_qb_txkd_qb_info
#  Lzid:		 
#  Git:          git@git.code.oa.com:txkd-data/qb.git
#  ParentTaskIds:
#  Description:  账号真实曝光
#  ModifyUser:   v_sxsun
#  ModifyDate:   2020-12-15
#  ModifyDesc:   创建文件
# ====================================================================================================================
time = __import__('time')
datetime = __import__('datetime')
string = __import__('string')

def setJobname(tdw, day):
    sql='use pcg_txkd_qb_info_app'
    tdw.execute(sql)

    sql = "SET mapred.job.name='t_app_qb_cidnum_real_exp_sum_d_%s';" % (day)
    tdw.execute(sql) 

def dohql(tdw, day):
    sql = 'use pcg_txkd_qb_info_app'
    tdw.execute(sql)

    sql = """
        INSERT OVERWRITE TABLE t_app_qb_cidnum_real_exp_sum_d PARTITION (p_date= %(day)s)
		select
			%(day)s as p_date
			,cid_num_range
			,overall_level
			,real_exp_text
			,real_exp_short_video
			,real_exp_small_video
			,real_exp
			,real_exp_total_text
			,real_exp_total_short_video
			,real_exp_total_small_video
			,real_exp_total
		from
		(
			select
				%(day)s as p_date
				,cid_num_range
				,overall_level
				,sum(real_exp_text) as real_exp_text
				,sum(real_exp_short_video) as real_exp_short_video
				,sum(real_exp_small_video) as real_exp_small_video
				,sum(real_exp) as real_exp
			from
			(
				select
					p_date
					,puin
					,case when overall_level=1 then '1'
						when overall_level=2 then '2'
						when overall_level>=3 and overall_level<=5 then '3~5'
						else '' end as overall_level
					,sum(case when busi_type='1' then real_exp end) as real_exp_text
					,sum(case when busi_type='3' then real_exp end) as real_exp_short_video
					,sum(case when busi_type='19' then real_exp end) as real_exp_small_video
					,sum(real_exp) as real_exp
				from
				pcg_txkd_qb_info_app::t_dwt_qb_content_c_user_indi_1d_d
				where p_date = %(day)s
				--and puin is not null
				group by p_date,puin
				,case when overall_level=1 then '1'
					when overall_level=2 then '2'
					when overall_level>=3 and overall_level<=5 then '3~5'
					else '' end
			)t1
			left join	
			(
				select
				t1.puin
				,case when nvl(cid_num,0)>=0 and nvl(cid_num,0)<=10 then '0~10'
					when cid_num>10 and cid_num<=30 then '10~30'
					when cid_num>30 and cid_num<=50 then '30~50'
					when cid_num>50 and cid_num<=100 then '50~100'
					when cid_num>100 then '100+' end as cid_num_range
				from				
				(
				select
					puin
				from
				t_dim_qb_content_info_1d_d
				where p_date = %(day)s
				and puin is not null
				group by puin
				)t1
				left join				
				(
					select
						puin
						,avg(cid_num) cid_num
					from
					(
					select
						from_unixtime(input_ts,'yyyyMMdd') as input_day
						,puin
						,count(distinct cid) as cid_num
					from
					pcg_txkd_shared_data_app::t_dim_fcc_b_article_rowkey_acc_d
					where p_date = %(day)s
					and from_unixtime(input_ts,'yyyyMMdd') between date_sub(%(day)s,6) and %(day)s
					and cid is not null
					and puin is not null
					group by from_unixtime(input_ts,'yyyyMMdd'),puin
					
					union all
					select
						from_unixtime(input_ts,'yyyyMMdd') as input_day
						,puin
						,count(distinct cid) as cid_num
					from
					pcg_txkd_shared_data_app::t_dim_fcc_b_video_rowkey_acc_d
					where p_date = %(day)s
					and from_unixtime(input_ts,'yyyyMMdd') between date_sub(%(day)s,6) and %(day)s
					and cid is not null
					and puin is not null
					group by from_unixtime(input_ts,'yyyyMMdd'),puin
					)
					group by puin
				) t2
				on t1.puin=t2.puin
			)t2
			on t1.puin=t2.puin
			group by
			cid_num_range
			,overall_level
		) r1
		left join
		(
			select 
				%(day)s as p_date
				,sum(case when busi_type=1 then real_exp_num end) as real_exp_total_text
				,sum(case when busi_type=3 then real_exp_num end) as real_exp_total_short_video
				,sum(case when busi_type=19 then real_exp_num end) as real_exp_total_small_video
				,sum(real_exp_num) as real_exp_total
			from
			pcg_txkd_qb_info_app::t_dwt_qb_real_exp_strict_aggr_d
			where p_date = %(day)s
		) r2
		on r1.p_date=r2.p_date
			
    """ % {'day':day}
    tdw.WriteLog (('execute sql:\n%s') % (sql))
    tdw.execute(sql)

def TDW_PL(tdw, argv=[]):
    day = argv[0]
    
    tdw.WriteLog('Start processing data %s' %day)
    setJobname(tdw, day)
    dohql(tdw, day)
    
    tdw.WriteLog ("all over")