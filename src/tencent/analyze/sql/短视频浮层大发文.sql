		select
			20210411 as p_date
			,cid_num_range 
			,overall_level 
			,product_name	
			,real_exp_short_video
			,real_exp_total_short_video
		from
		(
			select
				cid_num_range 
				,overall_level 
				,product_name
				,sum(real_exp_short_video) as real_exp_short_video  
			from
			(
				select
					puin
					,'短视频浮层' as product_name
					,sum(real_exp) as real_exp_short_video	--短视频浮层短视频阅读pv
				from
				pcg_txkd_qb_info_app::t_dwt_qb_content_c_user_indi_1d_d
				where p_date = 20210411 and product_id ='11' and busi_type ='3'
				group by 
					puin
				
			)t1
			join	
			( --发文等级&&发文量
			select 
				puin
				,'2' as overall_level
				,'10+' as cid_num_range	
			from
			sng_mp_etldata::t_dwt_union_puin_d
			where p_date=20210411 and overall_level=2 and metrics_7days_input>70
			group by puin
			)t2
			on t1.puin=t2.puin
			group by 
				cid_num_range 
				,overall_level 
				,product_name
		) r1
		left join
		(
            select
				sum(real_exp_num) as real_exp_total_short_video --总短视频浮层短视频阅读pv
			from
			pcg_txkd_qb_info_app::t_dwt_qb_real_exp_strict_aggr_d
			where p_date = 20210411  and product_id =11  and busi_type=3
		) r2
		on 1=1